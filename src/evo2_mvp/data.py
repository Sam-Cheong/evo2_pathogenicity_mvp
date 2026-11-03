"""Variant CSV loading and connected-gene held-out splits."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from .sequence import Variant


def load_variants(path: str | Path) -> list[Variant]:
    variants = []
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"chrom", "pos", "ref", "alt", "gene", "label"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}")
        for row in reader:
            value = row["label"].strip()
            if value not in ("", "0", "1"):
                raise ValueError(f"Invalid label: {value!r}")
            variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"],
                              row["gene"], None if value == "" else int(value))
            if variant.id in seen:
                raise ValueError(f"Duplicate variant in CSV: {variant.id}")
            if row.get("genome_build", "GRCh38") != "GRCh38":
                raise ValueError("Only GRCh38 CSV inputs are accepted")
            seen.add(variant.id)
            variants.append(variant)
    if not variants:
        raise ValueError("Variant CSV is empty")
    return variants


def genes_of(variant: Variant) -> set[str]:
    return {x.strip().upper() for x in re.split(r"[;,|]", variant.gene)
            if x.strip() and x.strip().upper() not in {".", "-", "NA", "N/A", "NONE", "UNKNOWN"}}


def _gene_groups(variants: list[Variant]) -> list[list[Variant]]:
    # Connected components avoid leakage for records annotated to overlapping genes.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for variant in variants:
        genes = sorted(genes_of(variant))
        if not genes:
            raise ValueError(f"Gene-held-out split requires a known gene: {variant.id}")
        for gene in genes[1:]:
            parent[find(gene)] = find(genes[0])
        find(genes[0])
    groups = {}
    for variant in variants:
        root = find(sorted(genes_of(variant))[0])
        groups.setdefault(root, []).append(variant)
    return list(groups.values())


def make_splits(variants: list[Variant], seed: int = 42, val_fraction: float = .2,
                test_fraction: float = .2) -> dict[str, list[str]]:
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if len({v.id for v in variants}) != len(variants):
        raise ValueError("Duplicate variants cannot be split")
    if any(v.label is None for v in variants):
        raise ValueError("Training splits require labels for every variant")
    groups = _gene_groups(variants)
    if len(groups) < 3:
        raise ValueError(f"Only {len(groups)} independent gene groups; cannot make gene-held-out train/val/test")
    rng = random.Random(seed)
    n = len(variants)
    targets = [1-val_fraction-test_fraction, val_fraction, test_fraction]
    best = None
    # Label-aware stratification uses labels only, never model or test performance.
    for _ in range(300):
        candidate = list(groups)
        rng.shuffle(candidate)
        splits = [[], [], []]
        # All partitions receive a group; remaining groups go to relative largest deficit.
        for i, group in enumerate(candidate):
            if i < 3:
                index = i
            else:
                index = min(range(3), key=lambda k: len(splits[k]) / (targets[k]*n))
            splits[index].extend(group)
        score = sum(abs(len(part)/n-target) for part, target in zip(splits, targets))
        missing = sum(len({v.label for v in part}) < 2 for part in splits)
        score += 10 * missing
        if best is None or score < best[0]:
            best = (score, splits)
    result = {name: sorted(v.id for v in part)
              for name, part in zip(("train", "val", "test"), best[1])}
    verify_splits(variants, result)
    return result


def verify_splits(variants: list[Variant], splits: dict[str, list[str]]) -> None:
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Splits must contain exactly train, val, test")
    lookup = {v.id: v for v in variants}
    if len(lookup) != len(variants):
        raise ValueError("Duplicate variants in dataset")
    seen, gene_sets = set(), []
    for name in ("train", "val", "test"):
        ids = splits[name]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or internally duplicated partition: {name}")
        if seen.intersection(ids):
            raise ValueError("A variant occurs in multiple partitions")
        if set(ids) - set(lookup):
            raise ValueError("Split contains unknown variant IDs")
        genes = set()
        for variant_id in ids:
            variant_genes = genes_of(lookup[variant_id])
            if not variant_genes:
                raise ValueError("Gene-held-out verification requires known genes")
            genes.update(variant_genes)
        if any(genes.intersection(old) for old in gene_sets):
            raise ValueError("Gene leakage between partitions")
        gene_sets.append(genes)
        seen.update(ids)
    if seen != set(lookup):
        raise ValueError("Splits do not cover every dataset variant")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--test-fraction", type=float, default=.2)
    args = parser.parse_args()
    variants = load_variants(args.variants)
    splits = make_splits(variants, args.seed, args.val_fraction, args.test_fraction)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(splits, indent=2), encoding="utf-8")
    report = {key: {"n": len(ids), "positive": sum(v.label == 1 for v in variants if v.id in set(ids)),
                    "gene_groups": len(_gene_groups([v for v in variants if v.id in set(ids)]))}
              for key, ids in splits.items()}
    print(json.dumps({"split": "connected_gene_held_out", "seed": args.seed, "partitions": report}, indent=2))


if __name__ == "__main__":
    main()
