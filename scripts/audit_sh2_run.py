"""Independently audit a completed cache-only two-stage SH2 experiment.

Read-only with respect to training, model, and data artifacts. The only output
is artifact_audit.json. No optimizer, fitting, model selection, or Evo 2 API is
run. The completed-Notebook gate precedes every clinical test archive read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.dont_write_bytecode = True
os.environ.setdefault("HF_HOME", str(PROJECT / "work" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

from evo2_mvp.data import load_variants
from evo2_mvp.metrics import binary_metrics, fingerprint, set_seed
from evo2_mvp.models import build_student
from evo2_mvp.two_stage_data import representation_partition, validate_output_archive
from evo2_mvp.two_stage_student import (
    PROBE_NAMES, constant_baseline, new_student, options, state_sha256,
    standardized_teacher, teacher_statistics,
)
from evo2_mvp.utils import save_json, sha256_file


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def same(actual, expected, location="value", rtol=1e-8, atol=1e-9):
    """Compare nested JSON metrics, including undefined metrics and matrices."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), location + ": dictionary keys differ")
        for key in expected:
            same(actual[key], expected[key], location + "." + str(key), rtol, atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), location + ": list length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{location}[{index}]", rtol, atol)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool), location + ": expected numeric value")
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)), location + ": nonfinite numeric value")
        require(math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol), location + ": numeric values differ")
    else:
        require(actual == expected, location + ": values differ")


def finite_tree(value, location):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), location + ": saved tensor contains NaN/Inf")
    elif isinstance(value, np.ndarray) and value.dtype.kind not in {"U", "S", "O"}:
        require(bool(np.isfinite(value).all()), location + ": array contains NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), location + ": saved scalar contains NaN/Inf")


def load_state(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(artifact, dict), str(path) + ": expected checkpoint dictionary")
    finite_tree(artifact, str(path))
    return artifact


def best_validation_row(history, key):
    """Reproduce the documented validation-only improvement tolerance."""
    require(bool(history), "Empty training history")
    selected, best = None, float("inf")
    for expected_epoch, row in enumerate(history, 1):
        require(int(row["epoch"]) == expected_epoch, "Training curve has missing/out-of-order epochs")
        value = float(row[key])
        require(math.isfinite(value), "Nonfinite validation loss in history")
        if value < best - 1e-8:
            selected, best = row, value
    return selected


def stripped_state(model_state, prefix):
    return {key.removeprefix(prefix): value for key, value in model_state.items() if key.startswith(prefix)}


def metrics_from_rows(rows, expected_ids, expected_labels, split, location):
    require(len(rows) == len(expected_ids), location + ": wrong prediction count")
    require([row["variant_id"] for row in rows] == list(expected_ids), location + ": prediction IDs/order differ")
    require(all(row["split"] == split for row in rows), location + ": wrong clinical split")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    require(np.array_equal(labels, np.asarray(expected_labels, dtype=int)), location + ": labels differ from locked probe archive")
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    logits = np.asarray([float(row["logit"]) for row in rows], dtype=float)
    require(bool(np.isfinite(logits).all()), location + ": nonfinite logits")
    sigmoid = np.exp(-np.logaddexp(0., -logits))
    require(bool(np.allclose(sigmoid, scores, rtol=1e-6, atol=1e-7)), location + ": scores do not match logits")
    return binary_metrics(labels, scores), logits, labels


def audit(root, run_root, notebook_status, report):
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    torch.set_num_threads(4)

    def passed(name, **facts):
        report["checks"].append({"name": name, "status": "passed", **facts})
        print("PASS " + name, flush=True)

    # Do not read probe_test or test result files until the actual Notebook run
    # has completed. A partially executed notebook is never successful evidence.
    execution = read_json(notebook_status)
    require(execution.get("status") == "completed", "Notebook execution is not completed: " + str(execution.get("status")))
    canonical_path, executed_path = Path(execution["notebook"]), Path(execution["executed_notebook"])
    canonical, executed = read_json(canonical_path), read_json(executed_path)
    require(len(canonical["cells"]) == len(executed["cells"]), "Executed Notebook has a different cell count")
    code_indices = []
    execution_counts = []
    for index, (source, result) in enumerate(zip(canonical["cells"], executed["cells"])):
        require(source["cell_type"] == result["cell_type"] and source.get("id") == result.get("id"), "Executed Notebook cell identity mismatch")
        require("".join(source.get("source", [])) == "".join(result.get("source", [])), "Executed Notebook source differs from current canonical Notebook")
        if source["cell_type"] == "code" and "".join(source.get("source", [])).strip():
            code_indices.append(index)
            require(isinstance(result.get("execution_count"), int) and result["execution_count"] > 0, "A code cell was not actually executed")
            require(not any(item.get("output_type") == "error" for item in result.get("outputs", [])), "Executed Notebook contains a cell error")
            execution_counts.append(result["execution_count"])
    require(execution.get("successful_cells") == code_indices, "Runner successful-cells record does not cover every actual code cell")
    require(execution_counts == sorted(set(execution_counts)), "Notebook code cell execution order is inconsistent")
    passed("notebook_executed_all_cells", code_cells=len(code_indices), notebook_sha256=sha256_file(canonical_path), executed_notebook_sha256=sha256_file(executed_path))

    identity = read_json(run_root / "notebook_input_identity.json")
    hashes = identity["files_sha256"]
    expected_identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    require(identity["identity"] == expected_identity, "Notebook input-identity checksum mismatch")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.name != ".env", "Input identity contains an unexpected external/secret path")
        require(path.is_file() and sha256_file(path) == expected, "Input source changed since execution: " + name)
    require(str(torch.__version__) == identity["torch"], "Audit Torch version differs from the completed experiment")
    passed("source_identity_unchanged", experiment_identity=identity["identity"], source_files=len(hashes), teacher_sha256=hashes["outputs/teacher/best_teacher.pt"])

    # Each recorded subprocess stage must be complete and its outputs immutable.
    required_stages = ("reference", "structural", "tests", "distill", "probes", "evaluate", "prediction", "prediction_reloaded")
    for stage in required_stages:
        value = read_json(run_root / (stage + ".stage.json"))
        require(value.get("status") == "completed" and value.get("returncode") == 0, "Incomplete subprocess stage: " + stage)
        require(value["signature"]["experiment_identity"] == identity["identity"], "Subprocess stage used a different source identity")
        arguments = value["signature"]["arguments"]
        require(not any("test_evo2.py" in str(arg) or str(arg) == "evo2_mvp.cache" for arg in arguments), "An online Evo 2 command entered the cache-only run")
        if stage in ("prediction", "prediction_reloaded"):
            require("--student" in arguments and "--teacher" not in arguments, "Independent prediction invoked a Teacher")
        for name, expected in value["artifacts_sha256"].items():
            require(sha256_file(root / name) == expected, "Stage artifact changed: " + name)
    junit = ET.parse(run_root / "junit.xml").getroot()
    require(not junit.findall(".//failure") and not junit.findall(".//error"), "Notebook pytest run contains failures/errors")
    live_api_cases = [case for case in junit.findall(".//testcase") if case.get("name") == "test_real_evo2_phase0"]
    require(bool(live_api_cases) and all(case.find("skipped") is not None for case in live_api_cases), "Live Evo 2 API integration was not demonstrably skipped")
    passed("completed_stages_and_zero_api_route", recorded_stages=list(required_stages), live_evo2_api_integration="skipped")

    data_dir, training = run_root / "data", run_root / "training"
    metadata_path = data_dir / "metadata.json"
    metadata, metadata_sha = read_json(metadata_path), sha256_file(metadata_path)
    require(metadata.get("status") == "complete" and metadata.get("synthetic") is False, "Prepared data are incomplete or synthetic")
    require(metadata["data_provenance"]["synthetic"] is False, "Data provenance is synthetic")
    source_paths = {"variants_sha256": "data/variants.csv", "splits_sha256": "data/splits.json", "fasta_sha256": "data/GRCh38.fa", "cache_manifest_sha256": "outputs/nvidia_cache/manifest.json"}
    for field, name in source_paths.items():
        require(metadata["source_hashes"][field] == hashes[name], "Prepared source identity mismatch: " + field)
    cache_manifest = read_json(root / "outputs/nvidia_cache/manifest.json")
    require(metadata["cache_metadata"] == cache_manifest["metadata"], "Teacher cache identity changed")
    require(metadata["cache_metadata"]["feature_dim"] == 24576 and metadata["cache_metadata"]["window"] == 256, "Wrong real Teacher embedding/window protocol")
    raw_splits = read_json(root / "data/splits.json")
    splits = raw_splits.get("splits", raw_splits)
    require(metadata["split_fingerprint"] == fingerprint(splits), "Clinical split identity differs")
    require(metadata["split_ids"] == splits, "Prepared outer ID allowlists differ from original splits")
    variants = load_variants(root / "data/variants.csv")
    partition = representation_partition(variants, splits, window=256, seed=int(metadata["random_seed"]))
    require(partition["train"] == metadata["stage1_train_ids"] and partition["val"] == metadata["stage1_val_ids"], "Representation split does not reproduce the label-independent gene/context algorithm")
    require(sorted(partition["excluded"]) == metadata["excluded_train_ids"], "Context-overlap exclusion report differs")
    require(set(metadata["teacher_feature_ids_read"]) == set(partition["retained"]) <= set(splits["train"]), "A clinical validation/test Teacher feature entered representation preparation")
    archives = {}
    for name, expected_ids in {"representation_train.npz": partition["train"], "representation_val.npz": partition["val"],
                               **{f"probe_{split}.npz": splits[split] for split in ("train", "val", "test")}}.items():
        path = data_dir / name
        require(sha256_file(path) == metadata["files_sha256"][name], "Prepared NPZ checksum mismatch: " + name)
        validate_output_archive(path, expected_ids, name.startswith("representation_"), window=256)
        with np.load(path, allow_pickle=False) as archive:
            archives[name] = {key: np.array(archive[key], copy=True) for key in archive.files}
        values = archives[name]
        changed = (values["ref"] != values["alt"]).any(axis=1)
        require(bool((changed.sum(axis=1) == 1).all() and changed[:, 128].all()), "An NPZ SNV is not located at the fixed variant center")
    lookup = {value.id: value for value in variants}
    for split in ("train", "val", "test"):
        require(np.array_equal(archives[f"probe_{split}.npz"]["labels"], [lookup[key].label for key in splits[split]]), "Probe label differs from original public ClinVar record")
    require(metadata["data_provenance"]["variant_label_fingerprint"] == fingerprint(sorted((v.id, v.label, v.gene) for v in variants)), "Legacy data provenance fingerprint differs")
    passed("sealed_data_and_label_isolation", counts=metadata["counts"], prepared_metadata_sha256=metadata_sha,
           representation_teacher_width=8192, clinical_teacher_features_read=False)

    import yaml
    config = yaml.safe_load((root / "configs/stripedhyena2_two_stage.yaml").read_text(encoding="utf-8"))
    distill_settings, probe_settings = options(config, "distillation"), options(config, "probe")
    initial = load_state(training / "initial_encoder.pt")
    best = load_state(training / "best_encoder.pt")
    last = load_state(training / "last_distillation.pt")
    distillation_status = read_json(training / "distillation_status.json")
    require(distillation_status.get("status") == "completed", "Representation distillation did not complete")
    require(distillation_status.get("labels_used") is False and distillation_status.get("clinical_validation_test_used") is False, "Stage 1 reports use of labels or clinical held-out data")
    protocol = {"configuration_sha256": fingerprint(config), "prepared_metadata_sha256": metadata_sha,
                "representation_files_sha256": {name: metadata["files_sha256"][name] for name in ("representation_train.npz", "representation_val.npz")},
                "train_ids_sha256": fingerprint(partition["train"]), "val_ids_sha256": fingerprint(partition["val"])}
    for name, value in (("initial", initial), ("best", best), ("last", last), ("status", distillation_status)):
        require(value["protocol"] == protocol, "Representation artifact protocol mismatch: " + name)
    initial_sha, best_sha = state_sha256(initial["encoder_state"]), state_sha256(best["encoder_state"])
    require(initial_sha == initial["encoder_sha256"] == distillation_status["initial_encoder_sha256"], "Initial encoder state checksum mismatch")
    require(best_sha == best["encoder_sha256"] == distillation_status["best_encoder_sha256"], "Best encoder state checksum mismatch")
    require(sha256_file(training / "best_encoder.pt") == distillation_status["best_encoder_file_sha256"], "Best encoder file checksum mismatch")
    require(initial_sha != best_sha, "Representation encoder did not actually update")
    require(best["initial_encoder_sha256"] == initial_sha and best["selected_by"] == "representation_validation_loss_only", "Best encoder initialization/selection provenance differs")
    set_seed(int(config.get("seed", 42)))
    initial_model, initial_kwargs = new_student(config, "linear")
    require(initial_kwargs == initial["model_kwargs"] and state_sha256(initial_model.encoder) == initial_sha, "Initial encoder is not the reproducible common seeded initialization")
    history = read_csv(training / "distillation_curve.csv")
    selected = best_validation_row(history, "val_total")
    require(int(selected["epoch"]) == best["best_epoch"] == distillation_status["best_epoch"] == last["best_epoch"], "Representation best_epoch is not selected solely by minimum representation-val loss")
    same(float(selected["val_total"]), best["best_validation"]["total"], "representation best loss")
    same(last["best_loss"], best["best_validation"]["total"], "resume best loss")
    require(len(history) == len(last["history"]) == distillation_status["epochs_run"] == last["epoch"], "Representation final/resume curve lengths differ")
    require(0 < len(history) <= distill_settings["epochs"], "Invalid representation training epoch count")
    for key, value in best["best_validation"].items():
        same(float(selected["val_" + key]), value, "selected representation validation." + key)
    rep_train = {key: torch.from_numpy(archives["representation_train.npz"][key]) for key in ("teacher_ref", "teacher_alt")}
    rep_val = {key: torch.from_numpy(archives["representation_val.npz"][key]) for key in ("teacher_ref", "teacher_alt")}
    statistics = teacher_statistics(rep_train, distill_settings["std_floor"])
    for key, value in statistics.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(best["teacher_statistics"][key], value, rtol=1e-6, atol=1e-7)
        else:
            require(best["teacher_statistics"][key] == value, "Teacher target scaler clamp metadata differs")
    baselines = {name: constant_baseline(standardized_teacher(values, statistics), distill_settings)
                 for name, values in (("train", rep_train), ("val", rep_val))}
    same(best["constant_mean_baseline"], baselines, "constant training-mean baseline", rtol=1e-6, atol=1e-7)
    require(float(best["best_validation"]["embedding_mean_variance"]) > 0, "Representation encoder outputs have collapsed to exactly constant embeddings")
    for name in ("projector_state", "delta_projector_state"):
        require(tuple(best[name]["weight"].shape) == (8192, config["model"]["d_model"]), "Wrong training-only projection dimensions")
    passed("stage1_real_representation_distillation", epochs=len(history), best_epoch=best["best_epoch"], initial_encoder_sha256=initial_sha,
           best_encoder_sha256=best_sha, best_validation=best["best_validation"], constant_mean_baseline=baselines,
           total_training_parameters=distillation_status["total_training_parameters"], training_only_projection_parameters=distillation_status["training_only_projection_parameters"])

    probes_status = read_json(training / "probes_status.json")
    require(probes_status.get("status") == "completed" and set(probes_status.get("completed", [])) == set(PROBE_NAMES), "Not all four probes completed")
    require(probes_status.get("all_probes_trained_before_test") is True, "Probe-before-test gate was not recorded")
    require(probes_status["prepared_metadata_sha256"] == metadata_sha and probes_status["configuration_sha256"] == fingerprint(config), "Probe experiment identity differs")
    checkpoints, computed_test, parameter_counts, individual_rows = {}, {}, {}, []
    expected_head_sha = {}
    for index, head_kind in enumerate(("linear", "mlp")):
        set_seed(int(config.get("seed", 42)) + 1000 + index)
        template, _ = new_student(config, head_kind)
        expected_head_sha[head_kind] = state_sha256(template.head)
    feature_caches = {}
    for arm in ("random", "distilled"):
        expected_encoder = initial_sha if arm == "random" else best_sha
        for split in ("train", "val", "test"):
            with np.load(training / "frozen_features" / f"{arm}_{split}.npz", allow_pickle=False) as archive:
                require(set(archive.files) == {"features", "variant_ids", "encoder_sha256", "source_sha256"}, "Unexpected frozen feature-cache schema")
                require(str(archive["encoder_sha256"].item()) == expected_encoder and str(archive["source_sha256"].item()) == metadata["files_sha256"][f"probe_{split}.npz"], "Frozen feature cache identity differs")
                require(archive["variant_ids"].tolist() == splits[split], "Frozen feature-cache IDs differ")
                values = np.array(archive["features"], copy=True)
                require(values.shape == (len(splits[split]), 3*config["model"]["d_model"]) and bool(np.isfinite(values).all()), "Malformed/nonfinite frozen features")
                feature_caches[(arm, split)] = torch.from_numpy(values)
    for name in PROBE_NAMES:
        directory = training / "probes" / name
        checkpoint_path = directory / "best_student.pt"
        checkpoint, complete = load_state(checkpoint_path), read_json(directory / "training_status.json")
        checkpoints[name] = checkpoint
        require(complete.get("status") == "completed" and complete.get("encoder_unchanged") is True and complete.get("clinical_test_opened") is False, "An incomplete/nonfrozen probe was promoted: " + name)
        checkpoint_sha = sha256_file(checkpoint_path)
        require(checkpoint_sha == complete["checkpoint_sha256"] == probes_status["checkpoints_sha256"][name], "Probe file checksum differs from training completion: " + name)
        require(checkpoint["kind"] == "student" and checkpoint["architecture"] == "stripedhyena2_tiny" and checkpoint["config"] == config, "Probe architecture/configuration differs")
        require(checkpoint["prepared_metadata_sha256"] == metadata_sha and checkpoint["split_fingerprint"] == metadata["split_fingerprint"], "Probe data lineage differs")
        require(checkpoint["data_provenance"] == metadata["data_provenance"] and checkpoint["cache_metadata"] == metadata["cache_metadata"], "Probe clinical/cache provenance differs")
        arm, head_kind = name.rsplit("_", 1)
        expected_encoder = initial_sha if arm == "random" else best_sha
        require(checkpoint["probe_arm"] == arm and checkpoint["head_kind"] == head_kind, "Probe arm/head identity differs")
        require(checkpoint.get("encoder_frozen") is True and checkpoint["encoder_sha256"] == expected_encoder, "Exported encoder is not the prescribed frozen state")
        require(checkpoint["common_initial_encoder_sha256"] == initial_sha and checkpoint["head_initial_sha256"] == expected_head_sha[head_kind], "Controlled initialization mismatch")
        require(state_sha256(stripped_state(checkpoint["model_state"], "encoder.")) == expected_encoder, "Probe encoder tensors changed during clinical head training")
        require(not {"projector_state", "delta_projector_state", "optimizer_state", "teacher_statistics"} & set(checkpoint), "Training-only projection/optimizer leaked into standalone inference artifact")
        require(not any("projector" in key for key in checkpoint["model_state"]), "Training-only projector is exported in model_state")
        model = build_student(checkpoint, "cpu")
        parameter_counts[name] = sum(p.numel() for p in model.parameters())
        require(parameter_counts[name] < 2_000_000, "Standalone student exceeds the <2M parameter limit")
        scaler = checkpoint["feature_scaler"]
        require(scaler["fit_split"] == "probe_train_only" and scaler["fit_variant_ids_sha256"] == fingerprint(splits["train"]), "Feature scaler fitted outside clinical training split")
        features = feature_caches[(arm, "train")]
        mean, scale = features.mean(0), features.std(0, unbiased=False).clamp_min(probe_settings["std_floor"])
        torch.testing.assert_close(checkpoint["model_state"]["feature_mean"], mean, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(checkpoint["model_state"]["feature_scale"], scale, rtol=1e-6, atol=1e-7)
        require(scaler["mean_sha256"] == state_sha256({"mean": mean}) and scaler["scale_sha256"] == state_sha256({"scale": scale}), "Probe scaler fingerprints do not reproduce train-only statistics")
        curve = read_csv(directory / "training_curve.csv")
        selected = best_validation_row(curve, "val_loss")
        require(int(selected["epoch"]) == checkpoint["best_epoch"] == complete["best_epoch"], "Clinical head best_epoch was not selected by validation BCE")
        require(len(curve) == checkpoint["n_epochs_run"] == complete["epochs_run"] and len(curve) <= probe_settings["epochs"], "Probe completed-epoch counts differ")
        require(checkpoint["selection"] == "minimum_clinical_validation_BCE_only" and checkpoint["test_used_for_selection"] is False, "Clinical test used for head selection")
        same(float(selected["val_loss"]), checkpoint["best_validation_loss"], name + ".best_validation_loss")
        validation_rows = read_csv(directory / "train_validation_predictions.csv")
        validation_metrics = read_json(directory / "validation_metrics.json")
        for split in ("train", "val"):
            selected_rows = [row for row in validation_rows if row["split"] == split]
            recomputed, logits, labels = metrics_from_rows(selected_rows, splits[split], archives[f"probe_{split}.npz"]["labels"], split, name + "." + split)
            same(validation_metrics[split], recomputed, name + "." + split + " metrics")
            if split == "val":
                bce = float((np.maximum(logits, 0.) - logits*labels + np.log1p(np.exp(-np.abs(logits)))).mean())
                same(bce, checkpoint["best_validation_loss"], name + ".validation BCE from saved logits", rtol=1e-6, atol=1e-7)
        test_rows = read_csv(directory / "test_predictions.csv")
        recomputed, _, _ = metrics_from_rows(test_rows, splits["test"], archives["probe_test.npz"]["labels"], "test", name + ".test")
        same(read_json(directory / "test_metrics.json"), recomputed, name + ".test metrics")
        test_provenance = read_json(directory / "test_provenance.json")
        require(test_provenance["checkpoint_sha256"] == checkpoint_sha and test_provenance["test_used_for_selection"] is False, "Test prediction lineage differs")
        computed_test[name] = recomputed
        individual_rows.extend({"probe": name, **row} for row in test_rows)
    comparison = read_json(training / "two_stage_test_comparison.json")
    require(comparison["test_used_for_selection"] is False and comparison["representation_training_ids_only"] is True, "Final comparison reports selection/test leakage")
    require(comparison["prepared_metadata_sha256"] == metadata_sha and comparison["test_ids_sha256"] == fingerprint(splits["test"]), "Final comparison test/data fingerprint differs")
    for name in PROBE_NAMES:
        same(comparison["models"][name]["test"], computed_test[name], name + ".comparison metrics")
    require(read_csv(training / "two_stage_test_predictions.csv") == individual_rows, "Combined test predictions differ from four locked individual exports")
    passed("four_frozen_probes_and_recomputed_test_metrics", inference_parameter_counts=parameter_counts,
           all_heads_controlled_same_initialization=True, encoder_states_unchanged=True, test_metrics=computed_test)

    saved_state_files = {}
    for path in sorted(run_root.rglob("*.pt")):
        load_state(path)  # Includes structural smoke and resume optimizer states.
        saved_state_files[path.relative_to(run_root).as_posix()] = sha256_file(path)
    require(bool(saved_state_files), "No saved tensor artifacts found")
    passed("all_saved_tensor_states_finite", checkpoint_files_sha256=saved_state_files)

    prediction, reloaded_prediction = read_json(run_root / "prediction.json"), read_json(run_root / "prediction_reloaded.json")
    require(prediction == reloaded_prediction, "Two independent Student CLI reload predictions differ")
    require(prediction["teacher_score"] is None and prediction["clinical_use"] == "research_only" and prediction["calibration_status"] == "not_calibrated", "Independent prediction used an unexpected Teacher/calibration path")
    finite_tree(prediction, "independent prediction")
    require(0 <= prediction["student_score"] <= 1 and prediction["student_predictive_entropy"] >= 0, "Invalid standalone score/entropy")
    run_report = read_json(run_root / "run_report.json")
    require(run_report["status"] == "completed" and run_report["experiment_identity"] == identity["identity"] and run_report["new_evo2_api_calls"] == 0, "Final Notebook report is incomplete or not cache-only")
    require(run_report["standalone_prediction"] == prediction and run_report["teacher_unchanged"] is True, "Final Notebook prediction/source report differs")
    for name in PROBE_NAMES:
        require(run_report["models"][name]["checkpoint_sha256"] == probes_status["checkpoints_sha256"][name], "Notebook summary checkpoint hash differs")
        require(run_report["models"][name]["parameters"] == parameter_counts[name], "Notebook inference parameter count differs")
    passed("independent_student_predictions_and_final_report", predictions_identical=True, clinical_use="research_only", new_evo2_api_route_used=False)
    report.update(status="passed", experiment_identity=identity["identity"], prepared_metadata_sha256=metadata_sha,
                  counts=metadata["counts"], inference_parameter_counts=parameter_counts,
                  independently_recomputed_test_metrics=computed_test,
                  methodology_limitations=["single seed", "historically examined clinical test cohort", "custom small SH2 reference variant; not official pretrained weights"],
                  clinical_use="research_only")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/sh2_two_stage_20261104")
    parser.add_argument("--notebook-status", default="outputs/sh2_notebook_execution.json")
    parser.add_argument("--output", help="Default: RUN_ROOT/artifact_audit.json")
    args = parser.parse_args()
    root = PROJECT.resolve()
    run_root = (root / args.run_root).resolve()
    output = (root / args.output).resolve() if args.output else run_root / "artifact_audit.json"
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_root": str(run_root), "audit_script_sha256": sha256_file(__file__),
              "training_performed_by_audit": False, "api_calls_performed_by_audit": 0, "checks": []}
    try:
        audit(root, run_root, (root / args.notebook_status).resolve(), report)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output, report)
        print(json.dumps({"status": "passed", "checks": len(report["checks"]), "output": str(output)}, indent=2))
    except BaseException as exc:
        report.update(status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat(),
                      error=type(exc).__name__ + ": " + str(exc))
        save_json(output, report)
        print("AUDIT FAILED: " + str(exc), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
