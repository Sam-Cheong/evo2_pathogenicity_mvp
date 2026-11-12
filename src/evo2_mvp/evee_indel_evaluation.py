"""Evaluate a frozen EVEE short-window probe on gene-held-out public ClinVar indels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import FileLock
import numpy as np
import torch

from .evee_covariance import EVEEStyleCovarianceProbe
from .evee_features import EVEEFeatureCache, INDEL_METHOD, extract_features
from .evee_indel import IndelVariant
from .evee_indel_cohort import prepare_indel_cohort
from .metrics import binary_metrics, write_csv
from .nvidia_encoder import NvidiaEvo2Encoder
from .utils import fingerprint, save_json, sha256_file


DEFAULT_SOURCE_RUN = Path("outputs/evee_full_2000_20261111/partial_20261112_181150")
MATCHED_PROTOCOL_FIELDS = (
    "model", "endpoint", "api_schema", "model_revision", "service_revision", "layer",
    "tensor_key", "tokenizer", "adapter_version", "genome_build", "reference_sha256",
    "window", "top_k", "feature_dim", "orientations", "delta",
)
INDEL_TYPES = ("insertion", "deletion")


def _load_source(source_run):
    cohort_path = source_run / "cohort.json"
    checkpoint_path = source_run / "probe" / "evee_covariance.pt"
    summary = json.loads((source_run / "run_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["cohort_sha256"] != sha256_file(cohort_path):
        raise ValueError("Source SNV run is incomplete or its cohort changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    split_fingerprint = fingerprint({
        name: [row["variant_id"] for row in partition] for name, partition in cohort["partitions"].items()
    })
    protocol = summary["feature_protocol"]
    if (checkpoint.get("artifact_kind") != "evee_style_covariance_probe"
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("split_fingerprint") != split_fingerprint
            or checkpoint["config"]["input_shape"] != [2, protocol["top_k"], protocol["feature_dim"]]):
        raise ValueError("Frozen probe does not match the source SNV run and feature shape")
    source = {"training_cohort_sha256": summary["cohort_sha256"],
              "checkpoint_sha256": sha256_file(checkpoint_path), "feature_protocol": protocol}
    return checkpoint, source


def _prepare(args, source):
    protocol = source["feature_protocol"]
    available = protocol["window"] - protocol["variant_index"] - protocol["top_k"]
    if args.max_indel_bases > available:
        raise ValueError(f"The source probe permits indels of at most {available} bases")
    cohort = prepare_indel_cohort(
        args.vcf, args.fasta, args.source_variants, args.source_run / "cohort.json",
        per_stratum=args.per_stratum, seed=args.seed, max_indel_bases=args.max_indel_bases,
        window=protocol["window"],
    )
    if cohort["reference_sha256"] != protocol["reference_sha256"]:
        raise ValueError("The indel cohort and trained probe use different reference FASTAs")
    cohort["frozen_snv_source"] = source
    cohort["cohort_fingerprint"] = fingerprint(cohort)
    path = args.output / "cohort.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != cohort:
        raise ValueError("Cohort differs from the existing run. Use a new output directory.")
    save_json(path, cohort)
    print(f"Prepared {len(cohort['records'])} held-out ClinVar indels.", flush=True)


def _load_cohort(output, source):
    cohort = json.loads((output / "cohort.json").read_text(encoding="utf-8"))
    expected_fingerprint = cohort.pop("cohort_fingerprint")
    if fingerprint(cohort) != expected_fingerprint or cohort["frozen_snv_source"] != source:
        raise ValueError("Cohort identity or frozen SNV source changed")
    records = cohort["records"]
    variants = [IndelVariant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"], row["label"])
                for row in records]
    if (not records or len({v.id for v in variants}) != len(records)
            or any(v.id != row["variant_id"] or v.label not in (0, 1)
                   or row["indel_type"] != ("insertion" if len(v.alt) > len(v.ref) else "deletion")
                   for v, row in zip(variants, records))):
        raise ValueError("Cohort must contain unique, labelled insertion/deletion records")
    return records, variants


def _extract(args, source, variants):
    protocol = source["feature_protocol"]
    reference_hash = sha256_file(args.fasta)
    if reference_hash != protocol["reference_sha256"]:
        raise ValueError("Reference FASTA differs from the source run")
    encoder_fields = ("model", "endpoint", "api_schema", "layer", "tensor_key", "service_revision")

    def encoder_factory():
        return NvidiaEvo2Encoder(**{key: protocol[key] for key in encoder_fields},
                                 timeout_seconds=45, poll_timeout_seconds=60, max_retries=1)

    def progress(_variant_id, counts):
        done = counts["processed"] + counts["skipped"]
        if done == 1 or done % 4 == 0 or done == len(variants):
            print(f"Indel features {done}/{len(variants)} ({counts['processed']} new)", flush=True)

    encoder = encoder_factory()
    try:
        stats = extract_features(encoder, variants, args.fasta, args.output / "features", indels=True,
                                 window=protocol["window"], top_k=protocol["top_k"],
                                 reference_sha256=reference_hash, progress=progress, workers=args.workers,
                                 encoder_factory=encoder_factory)
    finally:
        encoder.session.close()
    if stats["failed"] or stats["processed"] + stats["skipped"] != len(variants):
        raise RuntimeError("Indel feature extraction is incomplete")


@torch.inference_mode()
def _score(checkpoint, cache, variants, batch_size):
    probe = EVEEStyleCovarianceProbe(**checkpoint["model_kwargs"])
    probe.load_state_dict(checkpoint["model_state"], strict=True)
    probe.eval()
    mean_probe = checkpoint["mean_pooled"]
    mean_input = checkpoint["config"]["mean_pooled_input"]
    if mean_input not in {"mean_of_retained_positions", "full_downstream_mean"}:
        raise ValueError("Unknown frozen mean-pooled input protocol")

    scores = []
    for start in range(0, len(variants), batch_size):
        batch = [cache.get(v.id) for v in variants[start:start + batch_size]]
        delta = torch.from_numpy(np.stack([arrays["delta"] for arrays in batch]))
        if mean_input == "mean_of_retained_positions":
            pooled = delta.mean(dim=2)
        else:
            pooled = torch.from_numpy(np.stack([arrays["mean"] for arrays in batch]))
        scaled = (pooled.flatten(1).double() - mean_probe["mean"]) / mean_probe["scale"]
        covariance = probe(delta).double()
        mean_pooled = scaled @ mean_probe["weight"] + mean_probe["bias"]
        logits = torch.stack((covariance, mean_pooled), dim=1)
        if not torch.isfinite(logits).all():
            raise ValueError("Frozen probe produced nonfinite logits")
        scores.append(logits.sigmoid().numpy())
    return np.concatenate(scores)


def _evaluate(args, checkpoint, source, records, variants):
    cache = EVEEFeatureCache(args.output / "features")
    if cache.metadata.get("method") != INDEL_METHOD or any(
        cache.metadata.get(key) != source["feature_protocol"][key] for key in MATCHED_PROTOCOL_FIELDS
    ):
        raise ValueError("Indel cache uses a different encoder or feature configuration from the frozen probe")
    scores = _score(checkpoint, cache, variants, args.batch_size)
    labels = np.array([v.label for v in variants])
    kinds = np.array([row["indel_type"] for row in records])
    metrics = {name: binary_metrics(labels, scores[:, column])
               for column, name in enumerate(("covariance", "mean_pooled"))}
    by_type = {
        kind: {name: binary_metrics(labels[kinds == kind], scores[kinds == kind, column])
               for column, name in enumerate(("covariance", "mean_pooled"))}
        for kind in INDEL_TYPES if (kinds == kind).any()
    }
    predictions = [
        {"variant_id": row["variant_id"], "clinvar_variation_id": row["source_accession"],
         "gene": row["gene"], "indel_type": row["indel_type"], "label": row["label"],
         "covariance_score": float(covariance), "mean_pooled_score": float(mean_pooled)}
        for row, (covariance, mean_pooled) in zip(records, scores)
    ]
    prediction_path = args.output / "predictions.csv"
    write_csv(prediction_path, predictions)
    summary = {"status": "complete", "scope": "frozen_snv_probe_on_public_ClinVar_short_indels",
               "trained_on_indels": False, "clinical_use": "research_only",
               "cohort_sha256": sha256_file(args.output / "cohort.json"),
               "feature_manifest_sha256": sha256_file(args.output / "features" / "manifest.json"),
               "predictions_sha256": sha256_file(prediction_path), "source": source,
               "indel_feature_protocol": cache.metadata, "n_scored": len(scores),
               "n_expected": len(records), "test": metrics, "by_indel_type": by_type}
    save_json(args.output / "run_summary.json", summary)
    for name, result in metrics.items():
        print(f"{name}: n={result['n']}, AUROC={result['auroc']}, AP={result['auprc']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "evaluate", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--source-variants", type=Path, default=Path("data/variants.csv"))
    parser.add_argument("--vcf", type=Path, default=Path("data/clinvar.vcf.gz"))
    parser.add_argument("--fasta", type=Path, default=Path("data/GRCh38.fa"))
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--max-indel-bases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.per_stratum, args.max_indel_bases, args.batch_size) < 1:
        parser.error("--per-stratum, --max-indel-bases, and --batch-size must be positive")
    if args.output.resolve() == args.source_run.resolve():
        parser.error("Write the indel evaluation to a different directory from the SNV run")
    checkpoint, source = _load_source(args.source_run)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".evaluation.lock"), timeout=0):
        if args.phase in ("prepare", "run"):
            _prepare(args, source)
        if args.phase == "prepare":
            return
        records, variants = _load_cohort(args.output, source)
        if args.phase in ("extract", "run"):
            _extract(args, source, variants)
        if args.phase in ("evaluate", "run"):
            _evaluate(args, checkpoint, source, records, variants)


if __name__ == "__main__":
    main()
