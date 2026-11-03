"""Explicit binary metrics and reproducible, measured inference benchmarks."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def binary_metrics(labels, scores) -> dict:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError("Metrics require equally sized, nonempty one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Labels must be binary and scores finite probabilities")
    prediction = (scores >= 0.5).astype(int)
    both = len(np.unique(labels)) == 2
    has_positives = bool(labels.sum())
    has_positive_predictions = bool(prediction.sum())
    undefined = {}
    if not both:
        undefined.update({"auroc": "Only one true class is present", "auprc": "Only one true class is present"})
    if not has_positives:
        undefined.update({name: "No positive true labels are present" for name in ("precision", "recall", "f1")})
    elif not has_positive_predictions:
        undefined["precision"] = "No positive predictions are present"
    return {
        "n": int(labels.size), "positives": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if both else None,
        "auprc": float(average_precision_score(labels, scores)) if both else None,
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)) if has_positives and has_positive_predictions else None,
        "recall": float(recall_score(labels, prediction, zero_division=0)) if has_positives else None,
        "f1": float(f1_score(labels, prediction, zero_division=0)) if has_positives else None,
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "brier_score": float(brier_score_loss(labels, scores)),
        "classification_threshold": 0.5,
        "threshold_use": "metric_computation_only_not_a_clinical_cutoff",
        "undefined_metrics_reason": None if both else "Only one class is present; AUROC/AUPRC not reported",
        "undefined_metrics": undefined,
    }


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty results table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prediction_metadata(output, predictions, checkpoint_path, splits, provenance) -> dict:
    """Bind exported logits/labels to the exact saved model and split."""
    output = Path(output)
    metadata = {"schema_version": 1, "checkpoint_sha256": file_sha256(checkpoint_path),
                "predictions_sha256": file_sha256(output / "predictions.csv"),
                "data_provenance": provenance, "split_fingerprint": fingerprint(splits),
                "label_split_fingerprint": fingerprint(sorted((r["variant_id"], r["split"], int(r["label"])) for r in predictions)),
                "n_predictions": len(predictions)}
    write_json(output / "prediction_metadata.json", metadata)
    return metadata


def benchmark_callable(function, device, repeats=5, warmup=1) -> dict:
    """Measure inference wall time and process RSS; no fabricated hardware estimates."""
    import psutil
    device = torch.device(device)
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    durations = []
    rss_observed = rss_before
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        rss_observed = max(rss_observed, process.memory_info().rss)
    return {"device": str(device), "repeats": repeats, "warmup": warmup,
            "seconds_mean": float(np.mean(durations)), "seconds_std": float(np.std(durations)),
            "process_rss_before_bytes": rss_before, "process_rss_max_observed_bytes": rss_observed,
            "cpu_memory_measurement": "process RSS sampled between calls, not a transient peak",
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
