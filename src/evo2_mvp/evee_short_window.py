"""Run a short-window Evo 2 7B covariance-probe pilot on gene-held-out ClinVar SNVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from filelock import FileLock

from .data import _gene_groups, genes_of, verify_splits
from .evee_features import EVEEFeatureCache, extract_features, variant_contexts
from .nvidia_encoder import NvidiaEvo2Encoder
from .sequence import ReferenceGenome, Variant
from .train_teacher_head import choose_device, experiment_data
from .utils import save_json, sha256_file


MODEL = "arc/evo2-7b"
ENDPOINT = f"https://health.api.nvidia.com/v1/biology/{MODEL}/forward"
LAYER = "blocks.27"
PARTITIONS = ("train", "val", "test")
CLINVAR_LABELS = {
    0: {"Benign", "Likely_benign", "Benign/Likely_benign"},
    1: {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"},
}


def select_partition(variants, usable_ids, size, rng):
    """Select equal classes, with at most one SNV per connected gene group."""
    groups = []
    for component in _gene_groups(variants):
        name = "|".join(sorted(set().union(*(genes_of(v) for v in component))))
        candidates = sorted((v for v in component if v.id in usable_ids), key=lambda v: v.id)
        if candidates:
            groups.append((name, candidates))
    rng.shuffle(groups)
    half = size // 2
    # Use positive-only groups first so that mixed groups remain available for controls.
    positives = [group for group in groups if any(v.label == 1 for v in group[1])]
    positives.sort(key=lambda group: any(v.label == 0 for v in group[1]))
    positives = positives[:half]
    used = {name for name, _ in positives}
    negatives = [group for group in groups if group[0] not in used and any(v.label == 0 for v in group[1])][:half]
    if len(positives) != half or len(negatives) != half:
        raise ValueError(f"Too few independent ClinVar gene groups for {size} balanced SNVs")
    records = []
    for label, selected in ((1, positives), (0, negatives)):
        for group, candidates in selected:
            variant = rng.choice([v for v in candidates if v.label == label])
            records.append({"variant_id": variant.id, "chrom": variant.chrom, "pos": variant.pos,
                            "ref": variant.ref, "alt": variant.alt, "gene": variant.gene,
                            "group": group, "label": label})
    rng.shuffle(records)
    return records


def prepare_cohort(args, output):
    variants, source_splits, provenance = experiment_data(args.variants, args.splits)
    lookup = {v.id: v for v in variants}
    eligible = set()
    releases = set()
    with Path(args.variants).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") != "ClinVar" or row.get("synthetic", "").lower() != "false":
                raise ValueError("The pilot requires public, nonsynthetic ClinVar source records")
            variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"])
            if row.get("variant_id", variant.id) != variant.id:
                raise ValueError(f"ClinVar variant ID disagrees with its coordinates: {variant.id}")
            if int(row.get("review_stars") or 0) >= 1 and row.get("clinical_significance") in CLINVAR_LABELS[lookup[variant.id].label]:
                eligible.add(variant.id)
                releases.add(row.get("data_release_date", ""))
    if len(releases) != 1 or "" in releases:
        raise ValueError("The pilot requires one identified ClinVar release")

    usable = set()
    with ReferenceGenome(args.fasta) as genome:
        for variant in variants:
            if variant.id in eligible:
                try:
                    variant_contexts(genome, variant, args.window)
                except ValueError:
                    continue
                usable.add(variant.id)
    rng = random.Random(args.seed)
    sizes = dict(zip(PARTITIONS, args.sizes))
    partitions = {name: select_partition([lookup[key] for key in source_splits[name]], usable, sizes[name], rng)
                  for name in PARTITIONS}
    cohort = {"schema_version": 1, "seed": args.seed, "window": args.window, "top_k": args.top_k,
              "model": MODEL, "endpoint": ENDPOINT, "layer": LAYER, "genome_build": "GRCh38",
              "reference_sha256": sha256_file(args.fasta), "clinvar_release": releases.pop(),
              "variants_sha256": provenance["variants_sha256"], "source_splits_sha256": sha256_file(args.splits),
              "eligible_records": len(eligible), "valid_windows": len(usable),
              "selection": "balanced_within_existing_gene_holdout_one_snv_per_connected_group",
              "partitions": partitions}
    path = output / "cohort.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != cohort:
        raise ValueError("Cohort differs from the existing run. Use a new output directory.")
    save_json(path, cohort)
    print(f"Prepared {sum(sizes.values())} variants: {sizes}. Maximum {4 * sum(sizes.values())} forward requests before retries.",
          flush=True)
    return cohort


def load_cohort(output):
    cohort = json.loads((output / "cohort.json").read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "model": MODEL, "endpoint": ENDPOINT, "layer": LAYER, "genome_build": "GRCh38"}
    if any(cohort.get(key) != value for key, value in expected.items()):
        raise ValueError("Cohort does not identify the EVEE short-window 7B protocol")
    records = []
    indices = {}
    for name in PARTITIONS:
        partition = cohort["partitions"][name]
        indices[name] = list(range(len(records), len(records) + len(partition)))
        records.extend(partition)
    variants = [Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"], row["label"]) for row in records]
    if any(v.id != row["variant_id"] for v, row in zip(variants, records)):
        raise ValueError("Cohort record has inconsistent coordinates")
    verify_splits(variants, {name: [variants[i].id for i in indices[name]] for name in PARTITIONS})
    return cohort, records, variants, indices


class CachedArray:
    def __init__(self, cache, variants, field):
        self.cache = cache
        self.variants = variants
        self.field = field

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, index):
        return self.cache.get(self.variants[index].id)[self.field]


def run_extraction(args, output, cohort, variants):
    reference_hash = sha256_file(args.fasta)
    if reference_hash != cohort["reference_sha256"]:
        raise ValueError("FASTA differs from the prepared cohort")
    def make_encoder():
        return NvidiaEvo2Encoder(model=MODEL, endpoint=ENDPOINT, api_schema="nim_v1", layer=LAYER,
                                 service_revision=args.service_revision, timeout_seconds=45,
                                 poll_timeout_seconds=60, max_retries=1)

    def progress(variant_id, counts):
        done = counts["processed"] + counts["skipped"]
        if done == 1 or done % 4 == 0 or done == len(variants):
            print(f"Features {done}/{len(variants)} ({counts['processed']} new), latest {variant_id}", flush=True)

    stats = extract_features(make_encoder(), variants, args.fasta, output / "features", window=cohort["window"],
                             top_k=cohort["top_k"], reference_sha256=reference_hash, progress=progress,
                             workers=args.workers, encoder_factory=make_encoder)
    save_json(output / "extraction.json", stats)
    if stats.get("failed") or stats["processed"] + stats["skipped"] != len(variants):
        raise RuntimeError("Feature cache is incomplete. Rerun extract in the same directory to resume.")


def run_training(args, output, cohort, records, variants, indices):
    from .evee_covariance import fit_evee_covariance

    cache = EVEEFeatureCache(output / "features")
    for key in ("model", "endpoint", "layer", "window", "top_k", "reference_sha256", "genome_build"):
        if cache.metadata.get(key) != cohort[key]:
            raise ValueError(f"Feature cache differs from cohort for {key}")
    device = str(choose_device(args.device))
    summary_path = output / "run_summary.json"
    summary = {"status": "training", "scope": "balanced_short_window_ClinVar_SNVs_exploratory",
               "cohort_sha256": sha256_file(output / "cohort.json"),
               "feature_manifest_sha256": sha256_file(output / "features" / "manifest.json"),
               "feature_protocol": cache.metadata, "device": device}
    save_json(summary_path, summary)
    report = fit_evee_covariance(
        CachedArray(cache, variants, "delta"), [v.label for v in variants], [r["group"] for r in records],
        indices, means=None if args.matched_top_k_mean else CachedArray(cache, variants, "mean"),
        variant_ids=[v.id for v in variants],
        output=output / "probe", device=device, batch_size=args.batch_size, projection_dim=args.rank,
        epochs=args.epochs, seed=cohort["seed"], selection=args.selection,
    )
    summary.update(status="complete", split_counts=report["split_counts"],
                   test={name: scores["test"] for name, scores in report["metrics"].items()})
    save_json(summary_path, summary)
    print(json.dumps(summary["test"], indent=2, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "train", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=Path, default=Path("data/variants.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--fasta", type=Path, default=Path("data/GRCh38.fa"))
    parser.add_argument("--sizes", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"), default=(24, 8, 8))
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--service-revision", default="nvidia-hosted-7b-block27-v1")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--matched-top-k-mean", action="store_true",
                        help="Compute the mean baseline on the same retained positions as covariance")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--selection", choices=("last", "validation_bce"), default="last")
    args = parser.parse_args(argv)
    if args.phase in ("prepare", "run"):
        if any(size < 2 or size % 2 for size in args.sizes):
            parser.error("--sizes requires three positive even counts")
        if not 4 <= args.window <= 8192 or not 1 <= args.top_k <= args.window - 3 * args.window // 4:
            parser.error("--window must be 4–8192 and --top-k must fit the downstream positions")
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".pilot.lock"), timeout=0):
        if args.phase in ("prepare", "run"):
            prepare_cohort(args, args.output)
        if args.phase == "prepare":
            return
        cohort, records, variants, indices = load_cohort(args.output)
        if args.phase in ("extract", "run"):
            run_extraction(args, args.output, cohort, variants)
        if args.phase in ("train", "run"):
            run_training(args, args.output, cohort, records, variants, indices)


if __name__ == "__main__":
    main()
