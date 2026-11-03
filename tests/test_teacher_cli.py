"""Full head CLI smoke: synthetic fixtures and fake embeddings, no real claims."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from evo2_mvp.cache import FeatureCache
from evo2_mvp.data import load_variants, make_splits
from evo2_mvp.models import load_checkpoint


def test_teacher_cli_retrain_from_cache_and_reload(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    toy = tmp_path / "toy"
    result = subprocess.run([sys.executable, str(root / "scripts/create_toy_dataset.py"),
        "--out-dir", str(toy), "--n", "24"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    variants = load_variants(toy / "variants.csv")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(make_splits(variants)))
    cache = FeatureCache(tmp_path / "cache", {"encoder_type": "fake", "real_model_verified": False,
        "window": 256, "feature_dim": 6})
    for i, variant in enumerate(variants):
        # Deliberately arbitrary numerical fixtures, unrelated to pathogenicity.
        ref = np.array([i / 24, (i % 3) / 3], dtype=np.float32)
        alt = ref + np.array([.1, -.1], dtype=np.float32)
        delta = alt - ref
        cache.put(variant.id, {"ref": ref, "alt": alt, "delta": delta,
            "features": np.concatenate((ref, alt, delta))}, "ACGT", "AGGT")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"seed": 42, "teacher": {"hidden_dims": [8, 4], "dropout": 0.0},
        "training": {"epochs": 2, "batch_size": 8, "patience": 2}}))
    common = [sys.executable, "-m", "evo2_mvp.train_teacher_head", "--variants", str(toy / "variants.csv"),
        "--splits", str(splits_path), "--cache", str(cache.path), "--config", str(config), "--device", "cpu"]
    # Without the explicit guard, synthetic labels are forbidden.
    refused = subprocess.run(common + ["--output", str(tmp_path / "refused")], env=env, capture_output=True, text=True)
    assert refused.returncode != 0 and "Synthetic labels" in refused.stderr
    output = tmp_path / "head"
    trained = subprocess.run(common + ["--output", str(output), "--allow-synthetic-smoke-test"],
        env=env, capture_output=True, text=True)
    assert trained.returncode == 0, trained.stderr
    checkpoint = load_checkpoint(output / "best_teacher.pt")
    assert checkpoint["data_provenance"]["synthetic"]
    assert checkpoint["n_epochs_run"] == 2
    evaluated = tmp_path / "evaluated"
    reloaded = subprocess.run(common + ["--output", str(evaluated), "--allow-synthetic-smoke-test",
        "--evaluate", str(output / "best_teacher.pt")], env=env, capture_output=True, text=True)
    assert reloaded.returncode == 0, reloaded.stderr
    assert (output / "predictions.csv").read_text() == (evaluated / "predictions.csv").read_text()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["provenance"]["performance_scope"] == "synthetic_pipeline_smoke_test_only"
    assert (output / "training_curve.png").is_file()

