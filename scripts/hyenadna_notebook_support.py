"""Logged CLI execution and content-bound stage restart for the real notebook.

A stage is reusable only after a zero exit code AND validated complete artifacts.
This is stage-level restart; interrupted training restarts that model from its
fixed pretrained initialization, rather than claiming optimizer-state resume.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evo2_mvp.utils import save_json, sha256_file


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def input_identity(root):
    root = Path(root)
    files = ["configs/hyenadna_d256.yaml", "data/variants.csv", "data/splits.json",
             "outputs/teacher/best_teacher.pt", "outputs/nvidia_cache/manifest.json",
             "outputs/phase0_api_verified/result.json", "data/GRCh38.fa"]
    files += [p.relative_to(root).as_posix() for p in sorted((root / "src/evo2_mvp").rglob("*.py"))]
    files += ["scripts/prepare_hyenadna.py", "scripts/test_hyenadna.py",
              "scripts/compare_models.py", "scripts/hyenadna_notebook_support.py"]
    hashes = {name: sha256_file(root / name) for name in files}
    versions = {}
    from importlib.metadata import version
    for name in ("torch", "numpy", "transformers", "safetensors", "scikit-learn"):
        versions[name] = version(name)
    description = {"files_sha256": hashes, "versions": versions,
                   "python": sys.version, "device": "cpu", "cpu_threads": 4}
    identity = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, **description}


def validate_training(directory, mode, identity):
    from evo2_mvp.models import load_checkpoint
    from evo2_mvp.train_teacher_head import read_config
    from evo2_mvp.metrics import fingerprint
    directory = Path(directory)
    required = ["best_student.pt", "metrics.json", "predictions.csv", "prediction_metadata.json",
                "benchmark.json", "training_curve.csv", "training_curve.png", "config.json"]
    if mode == "distill":
        required.append("distillation_training_state.pt")
    for name in required:
        if not (directory / name).is_file():
            raise RuntimeError(f"Incomplete {mode} experiment: {directory / name}")
    checkpoint = load_checkpoint(directory / "best_student.pt")
    if checkpoint["mode"] != mode or checkpoint.get("architecture") != "hyenadna_d256":
        raise ValueError("Training mode/architecture mismatch")
    if checkpoint["config"] != read_config("configs/hyenadna_d256.yaml"):
        raise ValueError("Training configuration mismatch")
    if checkpoint["teacher_sha256"] != identity["files_sha256"]["outputs/teacher/best_teacher.pt"]:
        raise ValueError("Teacher lineage mismatch")
    if checkpoint["data_provenance"]["synthetic"] is not False:
        raise ValueError("Synthetic models cannot complete a real experiment")
    if checkpoint["data_provenance"]["variants_sha256"] != identity["files_sha256"]["data/variants.csv"]:
        raise ValueError("Dataset checksum mismatch")
    splits = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))
    if checkpoint["split_fingerprint"] != fingerprint(splits):
        raise ValueError("Split fingerprint mismatch")
    initialization = checkpoint["initialization"]
    if initialization.get("pretrained_encoder_loaded") is not True or initialization["parameter_count"] >= 2_000_000:
        raise ValueError("Expected a real pretrained student below 2M parameters")
    sidecar = json.loads((directory / "prediction_metadata.json").read_text(encoding="utf-8"))
    if sidecar["checkpoint_sha256"] != sha256_file(directory / "best_student.pt"):
        raise ValueError("Prediction checkpoint checksum mismatch")
    if sidecar["predictions_sha256"] != sha256_file(directory / "predictions.csv"):
        raise ValueError("Prediction CSV checksum mismatch")
    if checkpoint.get("n_epochs_run", 0) < 1 or checkpoint.get("best_epoch", 0) < 1:
        raise ValueError("No completed training epochs")
    return required


def run_stage(root, run_root, name, arguments, identity, artifacts, validate=None):
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / f"{name}.stage.json"
    signature = {"experiment_identity": identity["identity"], "arguments": list(arguments)}
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise RuntimeError(f"{name}: completed stage belongs to different inputs/code. Use a new RUN_ROOT.")
        for relative, checksum in previous["artifacts_sha256"].items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != checksum:
                raise RuntimeError(f"{name}: recorded artifact missing or modified: {relative}")
        if validate:
            validate()
        print(f"{name}: REUSED VERIFIED COMPLETED STAGE (no new training claimed)", flush=True)
        return previous
    log = run_root / f"{name}.log"
    if log.exists():
        log.rename(log.with_name(f"{name}.previous_{time.time_ns()}.log"))
    started = utc_now()
    command = [sys.executable, "-u", *map(str, arguments)]
    print("PowerShell: & " + subprocess.list2cmdline(command), flush=True)
    status = run_root / f"{name}.progress.json"
    save_json(status, {"status": "running", "started_at_utc": started, "arguments": list(arguments)})
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        with log.open("w", encoding="utf-8", buffering=1) as stream:
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        if validate:
            validate()
        hashes = {}
        for relative in artifacts:
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"{name} exited successfully but artifact missing: {path}")
            hashes[Path(relative).as_posix()] = sha256_file(path)
        result = {"status": "completed", "signature": signature, "started_at_utc": started,
                  "completed_at_utc": utc_now(), "returncode": 0, "artifacts_sha256": hashes}
        save_json(manifest, result)
        save_json(status, result)
        return result
    except BaseException as exc:
        save_json(status, {"status": "failed", "started_at_utc": started, "failed_at_utc": utc_now(),
                           "error": type(exc).__name__ + ": " + str(exc), "log": str(log)})
        raise
