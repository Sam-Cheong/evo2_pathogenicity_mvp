"""Research uncertainty measures; temperature is fitted on validation logits only."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def predictive_entropy(probability):
    """Bernoulli entropy in nats, not clinical confidence or disease-risk uncertainty."""
    value = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    safe = np.clip(value, 1e-15, 1 - 1e-15)
    result = -(safe * np.log(safe) + (1 - safe) * np.log1p(-safe))
    result = np.where((value == 0) | (value == 1), 0.0, result)
    return float(result) if result.ndim == 0 else result


def fit_temperature(validation_logits, validation_labels, *, split="val", max_iter=100) -> float:
    if split != "val":
        raise ValueError("Temperature calibration may only be fitted on the validation split")
    logits = torch.as_tensor(validation_logits, dtype=torch.float64).detach().flatten()
    labels = torch.as_tensor(validation_labels, dtype=torch.float64).detach().flatten()
    if logits.shape != labels.shape or logits.numel() < 2 or not torch.isfinite(logits).all():
        raise ValueError("Finite validation logits and matching labels are required")
    if not ((labels == 0) | (labels == 1)).all() or len(torch.unique(labels)) != 2:
        raise ValueError("Temperature fitting requires both validation classes")
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-5, 5).exp()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5, 5).exp())


def apply_temperature(logits, temperature: float):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return torch.as_tensor(logits) / temperature


def main(argv=None):
    import argparse
    import csv
    import json
    from pathlib import Path

    from .metrics import binary_metrics, file_sha256, fingerprint, write_json
    from .models import load_checkpoint

    parser = argparse.ArgumentParser(description="Fit temperature on validation predictions; assess once on untouched test predictions")
    parser.add_argument("--predictions", required=True, help="predictions.csv from this project's teacher/student evaluator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True, help="Output calibration JSON")
    args = parser.parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    splits = splits.get("splits", splits)
    if fingerprint(splits) != checkpoint["split_fingerprint"]:
        raise ValueError("Calibration splits differ from model training")
    with open(args.predictions, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    metadata_path = Path(args.predictions).parent / "prediction_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Predictions lack prediction_metadata.json; re-export with this project's evaluator")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"checkpoint_sha256": file_sha256(args.checkpoint),
                "predictions_sha256": file_sha256(args.predictions),
                "data_provenance": checkpoint["data_provenance"],
                "split_fingerprint": checkpoint["split_fingerprint"],
                "label_split_fingerprint": fingerprint(sorted((r["variant_id"], r["split"], int(r["label"])) for r in rows)),
                "n_predictions": len(rows)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Calibration prediction/model provenance mismatch: {key}")
    for split in ("train", "val", "test"):
        ids = [r["variant_id"] for r in rows if r["split"] == split]
        if len(ids) != len(set(ids)) or set(ids) != set(splits[split]):
            raise ValueError(f"Prediction records do not match the independent {split} split")
    if any(r["split"] not in ("train", "val", "test") for r in rows):
        raise ValueError("Prediction file contains an unknown split")
    values = {name: [r for r in rows if r["split"] == name] for name in ("val", "test")}
    validation_logits = [float(r["logit"]) for r in values["val"]]
    validation_labels = [int(r["label"]) for r in values["val"]]
    temperature = fit_temperature(validation_logits, validation_labels, split="val")
    test_logits = torch.tensor([float(r["logit"]) for r in values["test"]], dtype=torch.float64)
    test_labels = [int(r["label"]) for r in values["test"]]
    report = {"calibration_status": "validation_temperature_scaling", "fit_split": "val",
              "evaluation_split": "test", "temperature": temperature, "checkpoint_sha256": file_sha256(args.checkpoint),
              "split_fingerprint": checkpoint["split_fingerprint"], "model_kind": checkpoint["kind"],
              "data_provenance": checkpoint["data_provenance"], "predictions_sha256": file_sha256(args.predictions),
              "prediction_metadata_sha256": file_sha256(metadata_path),
              "test_uncalibrated": binary_metrics(test_labels, test_logits.sigmoid().numpy()),
              "test_calibrated": binary_metrics(test_labels, (test_logits / temperature).sigmoid().numpy()),
              "clinical_use": "research_only"}
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
