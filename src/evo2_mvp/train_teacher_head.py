"""Train/evaluate a cached-feature Evo 2 frozen-encoder classifier.

No encoder is loaded during head training. The cache must identify a genuinely
verified Evo 2 run; only an explicitly labelled synthetic smoke test can bypass
this check. Validation loss controls early stopping; test results do not.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cache import FeatureCache
from .data import load_variants, verify_splits
from .metrics import (benchmark_callable, binary_metrics, file_sha256, fingerprint,
                      set_seed, write_csv, write_json, write_prediction_metadata)
from .models import TeacherMLP, build_teacher, load_checkpoint
from .provenance import validate_evo2_metadata


def read_config(path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def experiment_data(variants_path, splits_path, allow_synthetic=False):
    variants = load_variants(variants_path)
    if not variants or any(v.label not in (0, 1) for v in variants):
        raise ValueError("Training requires nonempty variants with binary labels")
    with open(splits_path, encoding="utf-8") as stream:
        splits = json.load(stream)
    # The public file may wrap the split lists with provenance fields.
    if "splits" in splits:
        splits = splits["splits"]
    verify_splits(variants, splits)
    for split in ("train", "val", "test"):
        if not splits.get(split):
            raise ValueError(f"The {split} split is empty")
    with open(variants_path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    synthetic_flags = [str(r.get("source", r.get("data_source", ""))).strip().lower() == "synthetic"
                       or any(str(r.get(field, "")).strip().lower() in ("true", "1", "yes")
                              for field in ("synthetic", "is_synthetic")) for r in rows]
    if any(synthetic_flags) and not all(synthetic_flags):
        raise ValueError("Synthetic and actual variants must not be mixed for reported experiments")
    synthetic = bool(synthetic_flags and all(synthetic_flags))
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic labels are only allowed with --allow-synthetic-smoke-test")
    if allow_synthetic and not synthetic:
        raise ValueError("The synthetic bypass requires every CSV record to be marked synthetic")
    provenance = {"variants_sha256": file_sha256(variants_path),
                  "variant_label_fingerprint": fingerprint(sorted((v.id, v.label, v.gene) for v in variants)),
                  "n_variants": len(variants), "synthetic": synthetic,
                  "sources": sorted({str(r.get("source", r.get("data_source", "unspecified"))) for r in rows}),
                  "performance_scope": "synthetic_pipeline_smoke_test_only" if synthetic else "research_dataset"}
    return variants, splits, provenance


def check_cache(cache, provenance):
    metadata = cache.metadata
    if provenance["synthetic"]:
        return
    # The common validator distinguishes authentic local checkpoints from hosted
    # Evo 2 endpoints. Hosted services identify a run/revision rather than a
    # downloadable weight hash; neither backend can silently impersonate another.
    validate_evo2_metadata(metadata, require_verified=True)
    reference_hash = metadata.get("reference_sha256")
    if not isinstance(reference_hash, str) or len(reference_hash) != 64 or any(c not in "0123456789abcdef" for c in reference_hash.lower()):
        raise ValueError("Feature cache is missing its reference_sha256 provenance")
    if metadata.get("genome_build") != "GRCh38":
        raise ValueError("This MVP requires GRCh38 features")


def feature_tensors(variants, splits, cache):
    by_id = {v.id: v for v in variants}
    result = {}
    width = None
    for name in ("train", "val", "test"):
        arrays = [np.asarray(cache.get(key), dtype=np.float32) for key in splits[name]]
        if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
            raise ValueError("Cache contains malformed or nonfinite features")
        widths = {x.shape[0] for x in arrays}
        if len(widths) != 1:
            raise ValueError("Cache feature dimensions are inconsistent")
        current = next(iter(widths))
        if width is not None and current != width:
            raise ValueError("Cache feature dimensions differ between splits")
        width = current
        result[name] = (torch.from_numpy(np.stack(arrays)),
                        torch.tensor([by_id[key].label for key in splits[name]], dtype=torch.float32))
    if len(torch.unique(result["train"][1])) != 2:
        raise ValueError("Training split must contain both classes; regenerate a scientifically valid split")
    return result


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but PyTorch CUDA is unavailable")
    return device


def training_options(config):
    options = {"lr": 1e-3, "epochs": 20, "batch_size": 32, "patience": 5, "weight_decay": 0.01}
    options.update(config.get("training", {}))
    if options["epochs"] < 1 or options["batch_size"] < 1 or options["patience"] < 1 or options["lr"] <= 0:
        raise ValueError("Training epochs/batch size/patience/lr must be positive")
    return options


def fit_head(model, tensors, config, device):
    options = training_options(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(*tensors["train"]), batch_size=options["batch_size"], shuffle=True)
    model.to(device)
    model.fit_normalizer(tensors["train"][0].to(device))
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        total = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite teacher training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        model.eval()
        with torch.no_grad():
            validation_logits = model(tensors["val"][0].to(device))
            validation_loss = loss_fn(validation_logits, tensors["val"][1].to(device)).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite teacher validation loss")
        history.append({"epoch": epoch, "train_loss": total / len(tensors["train"][1]), "val_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= options["patience"]:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def report_predictions(model, tensors, splits, output, device, provenance, checkpoint_path):
    metrics, predictions = {}, []
    with torch.inference_mode():
        for name in ("train", "val", "test"):
            # Modest evaluation batches keep head/feature evaluation usable on CPU/GPU.
            logits = torch.cat([model(batch.to(device)).cpu() for batch in tensors[name][0].split(256)])
            scores = torch.sigmoid(logits).numpy()
            labels = tensors[name][1].numpy().astype(int)
            metrics[name] = binary_metrics(labels, scores)
            predictions.extend({"variant_id": key, "split": name, "label": int(label),
                                "logit": float(logit), "score": float(score)}
                               for key, label, logit, score in zip(splits[name], labels, logits.numpy(), scores))
    metrics["provenance"] = provenance
    write_json(Path(output) / "metrics.json", metrics)
    write_csv(Path(output) / "predictions.csv", predictions)
    write_prediction_metadata(output, predictions, checkpoint_path, splits, provenance)
    sample = tensors["test"][0][:1].to(device)
    with torch.inference_mode():
        benchmark = benchmark_callable(lambda: model(sample), device)
    benchmark["scope"] = "classification_head_only_cached_features_excludes_Evo2_encoder"
    write_json(Path(output) / "benchmark.json", benchmark)
    return metrics


def save_history(output, history):
    write_csv(Path(output) / "training_curve.csv", history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="train")
    ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "training_curve.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--allow-synthetic-smoke-test", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)
    variants, splits, provenance = experiment_data(args.variants, args.splits, args.allow_synthetic_smoke_test)
    cache = FeatureCache(args.cache)
    check_cache(cache, provenance)
    tensors = feature_tensors(variants, splits, cache)
    split_hash = fingerprint(splits)
    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate)
        if checkpoint["split_fingerprint"] != split_hash or checkpoint["data_provenance"] != provenance:
            raise ValueError("Evaluation data/split provenance differs from the training experiment")
        if checkpoint["cache_metadata"] != cache.metadata:
            raise ValueError("Evaluation cache differs from the trained teacher feature protocol")
        model = build_teacher(checkpoint, device)
    else:
        model_kwargs = {k: config.get("teacher", {}).get(k, v) for k, v in {"hidden_dims": [256, 64], "dropout": 0.2}.items()}
        model = TeacherMLP(tensors["train"][0].shape[1], **model_kwargs)
        model, history = fit_head(model, tensors, config, device)
        checkpoint = {"kind": "teacher", "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                      "input_dim": int(tensors["train"][0].shape[1]), "model_kwargs": model_kwargs,
                      "cache_metadata": cache.metadata, "seed": seed, "data_provenance": provenance,
                      "split_fingerprint": split_hash, "config": config, "clinical_use": "research_only",
                      "selection": "minimum_validation_BCE", "n_epochs_run": len(history)}
        torch.save(checkpoint, output / "best_teacher.pt")
        model = build_teacher(load_checkpoint(output / "best_teacher.pt"), device)
        save_history(output, history)
        write_json(output / "config.json", config)
    results = report_predictions(model, tensors, splits, output, device, provenance,
                                 args.evaluate or output / "best_teacher.pt")
    print(json.dumps({"status": "evaluated" if args.evaluate else "trained", "test": results["test"],
                      "performance_scope": provenance["performance_scope"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
