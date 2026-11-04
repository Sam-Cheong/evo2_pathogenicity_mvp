"""Synthetic fixtures verify stage isolation/frozen probes, not model accuracy."""
import builtins
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evo2_mvp import two_stage_student as pipeline


class FixtureStudent(nn.Module):
    def __init__(self, d_model=4, ffn_dim=8, num_heads=2, window=4, head_kind="linear", dropout=.1):
        super().__init__()
        self.encoder = nn.Conv1d(4, d_model, 3, padding=1)
        self.feature_dim, self.window = 3 * d_model, window
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.head = nn.Linear(self.feature_dim, 1) if head_kind == "linear" else nn.Sequential(
            nn.Linear(self.feature_dim, 8), nn.GELU(), nn.Dropout(dropout), nn.Linear(8, 1))

    def encode(self, value):
        return self.encoder(value)[:, :, value.shape[-1] // 2]

    def forward(self, ref, alt):
        r, a = self.encode(ref), self.encode(alt)
        fused = torch.cat([r, a, a-r], dim=-1)
        return self.head((fused-self.feature_mean)/self.feature_scale).squeeze(-1), fused


@pytest.fixture(autouse=True)
def test_model_and_threads(monkeypatch):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    def factory(config, head_kind):
        kwargs = pipeline.model_kwargs(config, head_kind)
        return FixtureStudent(**kwargs), kwargs
    monkeypatch.setattr(pipeline, "new_student", factory)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "data"; directory.mkdir()
    rng = np.random.default_rng(4)
    bases = rng.integers(0, 4, (16, 4)); bases[:, 2] = 0
    alternate = bases.copy(); alternate[:, 2] = 2
    ref = np.eye(4, dtype=np.float32)[bases].transpose(0, 2, 1)
    alt = np.eye(4, dtype=np.float32)[alternate].transpose(0, 2, 1)
    ids = np.asarray([f"1-{1000*(i+1)}-A-G" for i in range(16)])
    teacher_ref = rng.normal(size=(16, 6)).astype(np.float32)
    teacher_alt = teacher_ref + rng.normal(0, .2, (16, 6)).astype(np.float32)
    for name, sl in (("representation_train.npz", slice(0, 4)), ("representation_val.npz", slice(4, 6))):
        np.savez_compressed(directory/name, ref=ref[sl], alt=alt[sl], teacher_ref=teacher_ref[sl], teacher_alt=teacher_alt[sl], variant_ids=ids[sl])
    for name, sl in (("probe_train.npz", slice(0, 8)), ("probe_val.npz", slice(8, 12)), ("probe_test.npz", slice(12, 16))):
        np.savez_compressed(directory/name, ref=ref[sl], alt=alt[sl], labels=np.arange(16)[sl] % 2, variant_ids=ids[sl])
    splits = {"train": ids[:8].tolist(), "val": ids[8:12].tolist(), "test": ids[12:].tolist()}
    metadata = {"files_sha256": {path.name: pipeline.file_sha256(path) for path in directory.glob("*.npz")},
                "cache_metadata": {"window": 4, "feature_dim": 18, "reference_sha256": "a"*64,
                                   "feature_identity": "fixture_only", "genome_build": "GRCh38"},
                "data_provenance": {"synthetic": True, "n_variants": 16, "performance_scope": "unit_tests_only"},
                "split_ids": splits, "split_fingerprint": pipeline.fingerprint(splits),
                "stage1_train_ids": ids[:4].tolist(), "stage1_val_ids": ids[4:6].tolist()}
    pipeline.write_json(directory/"metadata.json", metadata)
    config = {"seed": 42, "cpu_threads": 1,
              "model": {"d_model": 4, "ffn_dim": 8, "num_heads": 2, "window": 4, "dropout": .1},
              "distillation": {"epochs": 2, "batch_size": 2, "eval_batch_size": 2, "patience": 2, "lr": .01},
              "probe": {"epochs": 2, "batch_size": 4, "feature_batch_size": 2, "patience": 2}}
    return directory, metadata, config


def forbid_files(monkeypatch, forbidden):
    real_open, real_io_open = builtins.open, io.open
    def guard(original):
        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).name in forbidden:
                raise AssertionError("A forbidden label/clinical-test file was opened: " + str(path))
            return original(path, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, "open", guard(real_open))
    monkeypatch.setattr(io, "open", guard(real_io_open))


def test_stage1_opens_only_unlabelled_representation_partitions(prepared, tmp_path, monkeypatch):
    directory, metadata, config = prepared
    output = tmp_path / "experiment"
    with monkeypatch.context() as guarded:
        forbid_files(guarded, {"probe_train.npz", "probe_val.npz", "probe_test.npz", "variants.csv", "best_teacher.pt"})
        best = pipeline.run_distillation(directory, output, config, "cpu")
    initial = pipeline.load_artifact(output/"initial_encoder.pt")
    assert "kind" not in initial and "kind" not in best
    assert best["encoder_sha256"] != initial["encoder_sha256"]
    assert best["initial_encoder_sha256"] == initial["encoder_sha256"]
    assert best["selected_by"] == "representation_validation_loss_only"
    with np.load(directory/"representation_train.npz") as data:
        expected_mean = np.concatenate([data["teacher_ref"], data["teacher_alt"]]).mean(0)
        expected_delta_mean = (data["teacher_alt"]-data["teacher_ref"]).mean(0)
    np.testing.assert_allclose(best["teacher_statistics"]["sequence_mean"].numpy(), expected_mean, atol=1e-6)
    np.testing.assert_allclose(best["teacher_statistics"]["delta_mean"].numpy(), expected_delta_mean, atol=1e-6)
    status = json.loads((output/"distillation_status.json").read_text())
    assert status["labels_used"] is False and status["clinical_validation_test_used"] is False
    assert status["projector_count"] == 2 and status["training_only_projection_parameters"] > 0
    assert status["constant_mean_baseline"]["val"]["total"] >= 0
    assert status["best_validation"]["embedding_mean_variance"] >= 0
    assert status["best_validation"]["delta_embedding_mean_std"] >= 0
    resumed = pipeline.run_distillation(directory, output, config, "cpu", resume=True)
    assert resumed["encoder_sha256"] == best["encoder_sha256"]
    changed = copy.deepcopy(config); changed["distillation"]["lr"] *= 2
    with pytest.raises(ValueError, match="Resume data/configuration"):
        pipeline.run_distillation(directory, output, changed, "cpu", resume=True)


def test_stage1_rejects_label_fields_and_clinical_ids(prepared, tmp_path):
    directory, metadata, config = prepared
    path = directory/"representation_train.npz"
    with np.load(path) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    np.savez_compressed(path, **values, labels=np.zeros(4))
    metadata["files_sha256"][path.name] = pipeline.file_sha256(path)
    pipeline.write_json(directory/"metadata.json", metadata)
    with pytest.raises(ValueError, match="no labels"):
        pipeline.run_distillation(directory, tmp_path/"forbidden", config, "cpu")
    np.savez_compressed(path, **values)
    metadata["files_sha256"][path.name] = pipeline.file_sha256(path)
    metadata["split_ids"]["train"].remove(values["variant_ids"][0])
    metadata["split_ids"]["test"].append(str(values["variant_ids"][0]))
    pipeline.write_json(directory/"metadata.json", metadata)
    with pytest.raises(ValueError, match="clinical validation/test"):
        pipeline.run_distillation(directory, tmp_path/"leakage", config, "cpu")


def test_four_frozen_probes_use_same_heads_train_scalers_and_no_test_read(prepared, tmp_path, monkeypatch):
    directory, metadata, config = prepared
    output = tmp_path/"experiment"
    pipeline.run_distillation(directory, output, config, "cpu")
    observed = []
    original = pipeline.verify_encoder_unchanged
    def observe(model, expected):
        original(model, expected)
        observed.append((model.encoder.training, any(p.grad is not None or p.requires_grad for p in model.encoder.parameters())))
    monkeypatch.setattr(pipeline, "verify_encoder_unchanged", observe)
    with monkeypatch.context() as guarded:
        forbid_files(guarded, {"probe_test.npz"})
        status = pipeline.run_probes(directory, output, config, "cpu", allow_synthetic=True)
    assert status["status"] == "completed" and status["all_probes_trained_before_test"] is True
    assert observed and all(item == (False, False) for item in observed)
    checkpoints = {name: pipeline.load_artifact(output/"probes"/name/"best_student.pt") for name in pipeline.PROBE_NAMES}
    for kind in ("linear", "mlp"):
        assert checkpoints["random_"+kind]["head_initial_sha256"] == checkpoints["distilled_"+kind]["head_initial_sha256"]
        assert checkpoints["random_"+kind]["common_initial_encoder_sha256"] == checkpoints["distilled_"+kind]["common_initial_encoder_sha256"]
    for name, checkpoint in checkpoints.items():
        model = FixtureStudent(**checkpoint["model_kwargs"]); model.load_state_dict(checkpoint["model_state"]); model.eval()
        encoder = {key.removeprefix("encoder."): value for key, value in checkpoint["model_state"].items() if key.startswith("encoder.")}
        assert pipeline.state_sha256(encoder) == checkpoint["encoder_sha256"]
        assert checkpoint["encoder_frozen"] is True and checkpoint["test_used_for_selection"] is False
        with np.load(output/"frozen_features"/(checkpoint["probe_arm"]+"_train.npz")) as archive:
            expected = torch.from_numpy(archive["features"].copy()).mean(0)
        assert torch.equal(model.feature_mean, expected)
        val = pipeline.load_partition(directory, "probe_val.npz", metadata, False)
        with torch.no_grad():
            direct, _ = model(val["ref"], val["alt"])
            r, a = model.encode(val["ref"]), model.encode(val["alt"])
            cached = pipeline.probe_logits(model, torch.cat([r, a, a-r], dim=-1), "cpu", 2)
        torch.testing.assert_close(direct, cached, rtol=1e-6, atol=1e-6)
    with monkeypatch.context() as guarded:
        forbid_files(guarded, {"probe_test.npz"})
        resumed = pipeline.run_probes(directory, output, config, "cpu", resume=True, allow_synthetic=True)
        assert resumed["status"] == "completed"


def test_evaluate_gate_precedes_test_read_and_reports_all_four(prepared, tmp_path, monkeypatch):
    directory, metadata, config = prepared
    output = tmp_path/"experiment"; output.mkdir()
    pipeline.write_json(output/"probes_status.json", {"status": "running", "completed": []})
    with monkeypatch.context() as guarded:
        forbid_files(guarded, {"probe_test.npz"})
        with pytest.raises(ValueError, match="All four"):
            pipeline.run_evaluation(directory, output, config, "cpu")
    (output/"probes_status.json").unlink()
    pipeline.run_distillation(directory, output, config, "cpu")
    pipeline.run_probes(directory, output, config, "cpu", allow_synthetic=True)
    report = pipeline.run_evaluation(directory, output, config, "cpu")
    assert set(report["models"]) == set(pipeline.PROBE_NAMES)
    assert report["test_used_for_selection"] is False
    for name, result in report["models"].items():
        metrics = json.loads((output/"probes"/name/"test_metrics.json").read_text())
        assert metrics == result["test"] and metrics["n"] == 4
    before = pipeline.file_sha256(output/"probes"/"random_linear"/"best_student.pt")
    pipeline.run_evaluation(directory, output, config, "cpu")
    assert pipeline.file_sha256(output/"probes"/"random_linear"/"best_student.pt") == before
