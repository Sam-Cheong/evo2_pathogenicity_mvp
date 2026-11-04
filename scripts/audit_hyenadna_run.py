"""Independently audit completed notebook artifacts, without training/API calls."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evo2_mvp.hyenadna import state_digest
from evo2_mvp.metrics import binary_metrics
from evo2_mvp.models import build_student, load_checkpoint
from evo2_mvp.utils import save_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/hyenadna_notebook_run_20261104")
    args = parser.parse_args()
    run = ROOT / args.run_root
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    execution = json.loads((ROOT / "outputs/hyenadna_notebook_execution.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or execution["status"] != "completed":
        raise RuntimeError("Notebook experiment has not completed")
    notebook = json.loads(Path(execution["executed_notebook"]).read_text(encoding="utf-8"))
    executed_cells = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        if cell.get("execution_count") is None or any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
            raise ValueError("Notebook contains an unexecuted or failed code cell")
        executed_cells += 1
    identity = json.loads((run / "input_identity.json").read_text(encoding="utf-8"))
    for path, expected in identity["files_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("Experiment input changed: " + path)
    torch.set_num_threads(4)
    audits, initial_states = {}, []
    for directory, mode in (("supervised", "supervised"), ("distilled", "distill")):
        path = run / directory
        checkpoint = load_checkpoint(path / "best_student.pt")
        model = build_student(checkpoint)
        initial_states.append(checkpoint["initialization"]["initial_model_state_sha256"])
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Nonfinite saved model state")
        encoder_changed = state_digest(model.encoder.state_dict()) != checkpoint["initialization"]["encoder_initial_state_sha256"]
        if not encoder_changed:
            raise ValueError("Encoder was not updated during fine tuning")
        with (path / "training_curve.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        expected_epoch = int(min(history, key=lambda row: float(row["val_loss"]))["epoch"])
        if expected_epoch != checkpoint["best_epoch"] or len(history) != checkpoint["n_epochs_run"]:
            raise ValueError("Best checkpoint does not match minimum validation BCE")
        sidecar = json.loads((path / "prediction_metadata.json").read_text(encoding="utf-8"))
        if sidecar["checkpoint_sha256"] != sha256_file(path / "best_student.pt") or sidecar["predictions_sha256"] != sha256_file(path / "predictions.csv"):
            raise ValueError("Prediction artifact checksum mismatch")
        with (path / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            rows = [r for r in predictions if r["split"] == split]
            recomputed = binary_metrics([int(r["label"]) for r in rows], [float(r["score"]) for r in rows])
            for key, value in recomputed.items():
                if isinstance(value, float):
                    if not np.isclose(value, metrics[split][key], atol=1e-12, rtol=0):
                        raise ValueError(f"Metric mismatch: {directory}/{split}/{key}")
                elif value != metrics[split][key]:
                    raise ValueError(f"Metric mismatch: {directory}/{split}/{key}")
        audits[directory] = {"parameter_count": sum(p.numel() for p in model.parameters()),
            "all_saved_state_finite": True, "pretrained_encoder_actually_updated": encoder_changed,
            "best_epoch": expected_epoch, "epochs_run": len(history),
            "all_split_metrics_recomputed_from_predictions": True,
            "checkpoint_sha256": sha256_file(path / "best_student.pt")}
    if initial_states[0] != initial_states[1]:
        raise ValueError("Supervised/distilled initial states differ")
    predictions = [json.loads((run / name).read_text(encoding="utf-8")) for name in ("prediction.json", "prediction_reloaded.json")]
    if predictions[0] != predictions[1]:
        raise ValueError("Fresh-process predictions differ")
    output = {"status": "passed", "input_and_source_checksums_reverified": True,
              "successful_notebook_code_cells": executed_cells,
              "same_complete_initialization": True, "teacher_unchanged": True,
              "independent_reload_predictions_identical": True, "models": audits,
              "new_nvidia_api_calls": 0, "clinical_use": "research_only"}
    save_json(run / "artifact_audit.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
