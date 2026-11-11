"""Compare mean and covariance on identical retained tokens as training data grows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

from filelock import FileLock
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import genes_of, load_variants
from .evee_covariance import fit_evee_covariance
from .evee_features import EVEEFeatureCache, variant_contexts
from .evee_short_window import load_cohort, select_partition
from .metrics import write_csv
from .sequence import ReferenceGenome
from .utils import save_json, sha256_file

SIZES = (12, 24, 40)
SEEDS = (42, 43, 44)


def prepare(previous, output):
    source, _, _, _ = load_cohort(previous)
    variants_path, splits_path = Path("data/variants.csv"), Path("data/splits.json")
    if (sha256_file(variants_path) != source["variants_sha256"]
            or sha256_file(splits_path) != source["source_splits_sha256"]):
        raise ValueError("Pilot source data changed")
    variants = {v.id: v for v in load_variants(variants_path)}
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    old = source["partitions"]
    partitions = {"train": list(old["train"]), "val": list(old["val"])}
    excluded = {g for part in old.values() for row in part for g in row["group"].split("|")}
    rng = random.Random(511)
    with ReferenceGenome("data/GRCh38.fa") as genome:
        for split, n in (("train", 16), ("test", 24)):
            part = [variants[k] for k in splits[split]]
            eligible = set()
            for v in part:
                if genes_of(v) & excluded:
                    continue
                try:
                    variant_contexts(genome, v, source["window"])
                except ValueError:
                    continue
                eligible.add(v.id)
            extra = select_partition(part, eligible, n, rng)
            partitions.setdefault(split, []).extend(extra)
    cohort = {**source, "selection": "extended_7b_pilot_excludes_previous_7b_test_genes",
              "partitions": partitions, "prior_cohort_sha256": sha256_file(previous/"cohort.json")}
    path = output/"cohort.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != cohort:
        raise ValueError("Existing learning-curve cohort differs")
    old_cache = EVEEFeatureCache(previous/"features")
    feature_dir = output/"features"
    feature_dir.mkdir(exist_ok=True)
    if not (feature_dir/"manifest.json").exists():
        records = {}
        for row in old["train"] + old["val"]:
            key = row["variant_id"]
            old_cache.get(key)
            records[key] = old_cache.manifest["records"][key]
            for direction in records[key]:
                cached_file = old_cache._path(key, direction)
                shutil.copyfile(cached_file, feature_dir/cached_file.name)
        save_json(feature_dir/"manifest.json", {**old_cache.manifest, "records": records})
    save_json(path, cohort)
    print("Prepared 40 train, 8 validation, 24 test. Reused 32 variants; 160 new forwards before retries.")


def nested_training(indices, labels, seed):
    rng = np.random.default_rng(seed)
    by_class = [rng.permutation([i for i in indices if labels[i] == k]) for k in (0, 1)]
    if any(len(ids) < max(SIZES)//2 for ids in by_class):
        raise ValueError("Too few training examples for the balanced curve")
    return {n: sorted(np.concatenate([ids[:n//2] for ids in by_class]).tolist()) for n in SIZES}


def pairwise_win_rates(scores, labels):
    positive, negative = scores[:, labels==1, None], scores[:, None, labels==0]
    return ((positive > negative) + 0.5*(positive == negative)).mean(axis=0)


def plot_curve(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [row["n_train"] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    left.plot(n, [row["mean_auroc"] for row in rows], "o-", label="Mean")
    left.plot(n, [row["covariance_auroc"] for row in rows], "s-", label="Covariance")
    left.axhline(.5, linestyle="--", linewidth=1, color=".65")
    left.set(ylabel="Test AUROC", ylim=(0, 1))
    left.legend(frameon=False)
    right.plot(n, [row["delta_auroc"] for row in rows], "o-", color="#6842a6")
    right.fill_between(n, [row["delta_ci_low"] for row in rows],
                       [row["delta_ci_high"] for row in rows], color="#6842a6", alpha=.19)
    right.axhline(0, linestyle="--", linewidth=1, color=".65")
    right.set(ylabel="AUROC difference (covariance − mean)")
    for ax in (left, right):
        ax.set(xlabel="Training SNVs", xticks=n)
    fig.savefig(output/"learning_curve.png", dpi=180)
    plt.close(fig)


def evaluate(output):
    torch.set_num_threads(4)
    cohort, records, variants, partitions = load_cohort(output)
    cache = EVEEFeatureCache(output/"features")
    for key in ("model", "endpoint", "layer", "window", "top_k", "reference_sha256"):
        if cache.metadata[key] != cohort[key]:
            raise ValueError(f"Unexpected cache protocol: {key}")
    arrays = [cache.get(v.id)["delta"] for v in variants]
    labels = [v.label for v in variants]
    groups = [r["group"] for r in records]
    test = partitions["test"]
    test_ids = [variants[i].id for i in test]
    y = np.array([labels[i] for i in test])
    if len({groups[i] for i in test}) != len(test):
        raise ValueError("Expected one test SNV per connected gene group")
    rng = np.random.default_rng(511)
    neg, pos = [rng.integers(count, size=(2000, count)) for count in
                (int((y == 0).sum()), int((y == 1).sum()))]
    details, curve, bootstrap_curves = [], [], []
    for n in SIZES:
        predictions = {"mean_pooled": [], "covariance": []}
        for seed in SEEDS:
            subset = nested_training(partitions["train"], labels, seed)[n]
            chosen = subset + partitions["val"] + test
            bounds = (0, n, n+len(partitions["val"]), len(chosen))
            splits = {name: list(range(a, b)) for name, a, b in zip(
                ("train", "val", "test"), bounds, bounds[1:])}
            result = fit_evee_covariance(
                [arrays[i] for i in chosen], [labels[i] for i in chosen], [groups[i] for i in chosen],
                splits, variant_ids=[variants[i].id for i in chosen], device="cpu",
                projection_dim=16, batch_size=4, epochs=240//n, learning_rate=.001, seed=seed)
            test_predictions = [row for row in result["predictions"] if row["split"] == "test"]
            if [row["variant_id"] for row in test_predictions] != test_ids:
                raise ValueError("Test prediction order differs from the fixed cohort")
            for name in predictions:
                predictions[name].append(np.array([row[name+"_score"] for row in test_predictions]))
            details.append({"n_train": n, "seed": seed, "train_ids": [variants[i].id for i in subset],
                            "metrics": result["metrics"], "predictions": test_predictions})
        mean, cov = (np.stack(predictions[name]) for name in ("mean_pooled", "covariance"))
        # Average per-seed AUCs. Resample the same test gene groups for both heads and all seeds.
        cov_auc, mean_auc = [np.mean([roc_auc_score(y, row) for row in scores]) for scores in (cov, mean)]
        differences = pairwise_win_rates(cov, y) - pairwise_win_rates(mean, y)
        deltas = differences[pos[:, :, None], neg[:, None, :]].mean(axis=(1, 2))
        bootstrap_curves.append(deltas)
        low, high = np.quantile(deltas, [.025, .975])
        curve.append({"n_train": n, "n_test": len(y), "mean_auroc": float(mean_auc),
                      "covariance_auroc": float(cov_auc), "delta_auroc": float(cov_auc-mean_auc),
                      "delta_ci_low": float(low), "delta_ci_high": float(high),
                      "mean_auprc": float(np.mean([average_precision_score(y, a) for a in mean])),
                      "covariance_auprc": float(np.mean([average_precision_score(y, a) for a in cov]))})
    trend_ci = np.quantile(bootstrap_curves[-1]-bootstrap_curves[0], [.025, .975])
    plot_curve(output, curve)
    write_csv(output/"learning_curve.csv", curve)
    save_json(output/"learning_curve.json", {"scope": "exploratory_7b_fixed_window_historical_partition",
              "seeds": list(SEEDS), "cohort_sha256": sha256_file(output/"cohort.json"),
              "feature_manifest_sha256": sha256_file(cache.manifest_path),
              "both_heads_use": "same_bidirectional_top_k_deltas",
              "training_sizes": list(SIZES), "covariance_updates": 60, "projection_dim": 16,
              "test_gene_bootstrap_replicates": 2000,
              "change_in_delta_auroc": curve[-1]["delta_auroc"]-curve[0]["delta_auroc"],
              "change_in_delta_auroc_95_ci": trend_ci.tolist(),
              "curve": curve, "runs": details})
    print(json.dumps(curve, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/evee_short_window_20261111"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output/".pilot.lock"), timeout=0):
        if args.phase == "prepare":
            prepare(args.previous, args.output)
        else:
            evaluate(args.output)


if __name__ == "__main__":
    main()
