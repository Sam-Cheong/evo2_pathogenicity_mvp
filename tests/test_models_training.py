"""Weight-free tests. These synthetic tensors do not measure pathogenicity."""
import math
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from evo2_mvp.calibration import fit_temperature, predictive_entropy
from evo2_mvp.metrics import binary_metrics
from evo2_mvp.models import TeacherMLP, TinyStudent, build_student, build_teacher, load_checkpoint, one_hot


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_teacher_backward_and_train_only_standardization():
    model = TeacherMLP(6, hidden_dims=(8, 4), dropout=0)
    training = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    model.fit_normalizer(training)
    assert torch.equal(model.input_mean, training.mean(0))
    original_mean = model.input_mean.clone()
    logits = model(torch.full((3, 6), 999.0))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(model.input_mean, original_mean)


def test_teacher_safe_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = TeacherMLP(6, hidden_dims=(8, 4), dropout=0).eval()
    data = torch.randn(3, 6)
    model.fit_normalizer(data)
    checkpoint = {"kind": "teacher", "input_dim": 6,
                  "model_kwargs": {"hidden_dims": [8, 4], "dropout": 0}, "model_state": model.state_dict()}
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint, path)
    loaded = build_teacher(load_checkpoint(path))
    assert torch.equal(model(data), loaded(data))


def test_teacher_training_selects_and_reloads_validation_state():
    from evo2_mvp.train_teacher_head import fit_head
    torch.manual_seed(3)
    features = torch.randn(16, 6)
    labels = torch.tensor([0., 1.] * 8)
    tensors = {"train": (features[:8], labels[:8]), "val": (features[8:12], labels[8:12]),
               "test": (features[12:], labels[12:])}
    model, history = fit_head(TeacherMLP(6, hidden_dims=(8, 4), dropout=0), tensors,
                              {"training": {"epochs": 2, "batch_size": 4, "patience": 2}}, torch.device("cpu"))
    with torch.no_grad():
        restored_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tensors["val"][0]), tensors["val"][1]).item()
    assert restored_loss == pytest.approx(min(r["val_loss"] for r in history), abs=1e-6)
    assert torch.isfinite(model(tensors["test"][0])).all()


def test_student_shared_encoder_backward_under_2m():
    student = TinyStudent()
    assert sum(p.numel() for p in student.parameters()) < 2_000_000
    ref, alt = one_hot(["ACGT" * 16, "TGCA" * 16]), one_hot(["AGGT" + "ACGT" * 15, "TGCA" * 16])
    logits, features = student(ref, alt)
    assert logits.shape == (2,)
    assert features.shape == (2, student.feature_dim)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert student.encoder[0].weight.grad is not None
    assert torch.isfinite(logits).all()
    with torch.no_grad():
        _, identical = student(ref, ref)
    assert torch.equal(identical[:, 2 * student.feature_dim // 3:], torch.zeros_like(identical[:, 2 * student.feature_dim // 3:]))


def test_student_checkpoint_has_no_teacher_dependency(tmp_path):
    student = TinyStudent(embedding_dim=8, channels=(4, 8), dropout=0).eval()
    checkpoint = {"kind": "student", "model_kwargs": {"embedding_dim": 8, "channels": [4, 8], "dropout": 0},
                  "model_state": student.state_dict()}
    path = tmp_path / "student.pt"
    torch.save(checkpoint, path)
    restored = build_student(load_checkpoint(path))
    ref, alt = one_hot(["ACGT" * 8]), one_hot(["AGGT" + "ACGT" * 7])
    assert torch.equal(student(ref, alt)[0], restored(ref, alt)[0])
    code = "import sys; from evo2_mvp.models import TinyStudent, one_hot; m=TinyStudent(); m(one_hot(['ACGT']),one_hot(['AGGT'])); assert 'evo2' not in sys.modules"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90, env=environment)
    assert result.returncode == 0, result.stderr


def test_binary_kd_detaches_teacher_and_temperature():
    from evo2_mvp.distill_student import binary_kd_loss
    teacher = torch.tensor([1., -2.], requires_grad=True)
    student = torch.tensor([0.2, -0.3], requires_grad=True)
    loss = binary_kd_loss(student, teacher, 2)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    with pytest.raises(ValueError):
        binary_kd_loss(student, teacher, 0)


def test_entropy_and_calibration_guard():
    assert predictive_entropy(0.5) == pytest.approx(math.log(2))
    assert predictive_entropy(0.) == 0.
    assert predictive_entropy(1.) == 0.
    with pytest.raises(ValueError):
        predictive_entropy(float("nan"))
    with pytest.raises(ValueError):
        fit_temperature([1., -1.], [1., 0.], split="test")
    temperature = fit_temperature([5., 1., -1., -5.], [1., 0., 1., 0.], max_iter=10)
    assert math.isfinite(temperature) and temperature > 0


def test_single_class_metrics_are_explicit():
    values = binary_metrics([1, 1], [0.2, 0.8])
    assert values["auroc"] is None and values["auprc"] is None
    assert values["confusion_matrix"] == [[0, 0], [1, 1]]
    assert values["brier_score"] == pytest.approx(0.34)
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2, np.nan])
    negatives = binary_metrics([0, 0], [0.2, 0.8])
    assert negatives["precision"] is None and negatives["recall"] is None and negatives["f1"] is None
    assert negatives["accuracy"] == 0.5 and negatives["brier_score"] == pytest.approx(.34)
    assert "recall" in negatives["undefined_metrics"]


def test_calibration_rejects_predictions_from_another_checkpoint(tmp_path):
    from evo2_mvp.calibration import main
    from evo2_mvp.metrics import fingerprint, write_csv, write_prediction_metadata
    splits = {"train": ["a", "b"], "val": ["c", "d"], "test": ["e", "f"]}
    provenance = {"synthetic": True, "performance_scope": "unit_test_only"}
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"kind": "teacher", "split_fingerprint": fingerprint(splits), "data_provenance": provenance}, checkpoint_path)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))
    rows = [{"variant_id": variant_id, "split": name, "label": label, "logit": float(2 * label - 1)}
            for name, ids in splits.items() for label, variant_id in enumerate(ids)]
    write_csv(tmp_path / "predictions.csv", rows)
    write_prediction_metadata(tmp_path, rows, checkpoint_path, splits, provenance)
    args = ["--predictions", str(tmp_path / "predictions.csv"), "--checkpoint", str(checkpoint_path),
            "--splits", str(splits_path), "--output", str(tmp_path / "calibration.json")]
    main(args)
    assert json.loads((tmp_path / "calibration.json").read_text())["fit_split"] == "val"
    # The unchanged logit CSV cannot silently calibrate a different saved model.
    torch.save({"kind": "teacher", "split_fingerprint": fingerprint(splits), "data_provenance": provenance,
                "different_model": True}, checkpoint_path)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        main(args)


def test_synthetic_column_cannot_bypass_training_guard(tmp_path):
    from evo2_mvp.metrics import write_csv
    from evo2_mvp.train_teacher_head import experiment_data
    rows = [{"chrom": "1", "pos": 100 + i, "ref": "A", "alt": "G", "gene": f"GENE{i}",
             "label": i % 2, "source": "custom", "synthetic": "true"} for i in range(3)]
    variants_path = tmp_path / "variants.csv"
    write_csv(variants_path, rows)
    splits = {name: [f"1-{100 + i}-A-G"] for i, name in enumerate(("train", "val", "test"))}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="Synthetic labels"):
        experiment_data(variants_path, splits_path)
    _, _, provenance = experiment_data(variants_path, splits_path, allow_synthetic=True)
    assert provenance["synthetic"] is True
    assert provenance["performance_scope"] == "synthetic_pipeline_smoke_test_only"
