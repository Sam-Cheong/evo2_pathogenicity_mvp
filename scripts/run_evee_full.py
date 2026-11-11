"""Prepare, resume, and train the fixed-window 2,000-SNV EVEE experiment."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

from filelock import FileLock
import torch

from evo2_mvp.data import _gene_groups, genes_of
from evo2_mvp.evee_features import EVEEFeatureCache, variant_contexts
from evo2_mvp.evee_short_window import load_cohort, main as run_phase
from evo2_mvp.nvidia_encoder import NvidiaAPIError
from evo2_mvp.sequence import ReferenceGenome
from evo2_mvp.train_teacher_head import experiment_data
from evo2_mvp.utils import save_json, sha256_file


OUTPUT = Path("outputs/evee_full_2000_20261111")
PILOTS = [Path("outputs") / name for name in (
    "evee_short_window_20261111", "evee_learning_curve_20261111",
)]


def prepare(output):
    previous, _, _, _ = load_cohort(PILOTS[0])
    variants, splits, provenance = experiment_data("data/variants.csv", "data/splits.json")
    if (len(variants) != 2000 or provenance["sources"] != ["ClinVar"]
            or provenance["variants_sha256"] != previous["variants_sha256"]
            or sha256_file("data/splits.json") != previous["source_splits_sha256"]):
        raise ValueError("The 2,000-SNV source cohort or its split has changed")
    with ReferenceGenome("data/GRCh38.fa") as genome:
        for variant in variants:
            variant_contexts(genome, variant, previous["window"])
    group_by_id = {}
    for component in _gene_groups(variants):
        group = "|".join(sorted({g for v in component for g in genes_of(v)}))
        group_by_id.update({v.id: group for v in component})
    records = {v.id: {"variant_id": v.id, "chrom": v.chrom, "pos": v.pos, "ref": v.ref,
                      "alt": v.alt, "gene": v.gene, "group": group_by_id[v.id], "label": v.label}
               for v in variants}
    cohort = {**previous, "selection": "all_original_clinvar_snvs_historical_gene_split",
              "partitions": {name: [records[key] for key in ids] for name, ids in splits.items()}}
    path = output/"cohort.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != cohort:
        raise ValueError("Existing run uses a different cohort")
    features = output/"features"
    features.mkdir(exist_ok=True)
    if not (features/"manifest.json").exists():
        caches = [EVEEFeatureCache(p/"features") for p in PILOTS]
        if caches[0].metadata != caches[1].metadata:
            raise ValueError("Pilot caches have different feature identities")
        reused = {}
        for cache in caches:
            for key, directions in cache.manifest["records"].items():
                if key not in records or key in reused:
                    continue
                cache.get(key)
                for direction in directions:
                    source = cache._path(key, direction)
                    shutil.copyfile(source, features/source.name)
                reused[key] = directions
        save_json(features/"manifest.json", {**caches[0].manifest, "records": reused})
    save_json(path, cohort)
    print("Prepared all 2,000 SNVs with the original gene-held-out splits", flush=True)


def status(output, state, **details):
    save_json(output/"job_status.json", {"state": state,
              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), **details})


def run(output):
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    common = {"started_at": started, "total_variants": 2000}
    try:
        for attempt in range(1, 13):
            status(output, "extracting", attempt=attempt, **common)
            try:
                run_phase(["extract", "--output", str(output), "--workers", "4"])
                break
            except NvidiaAPIError as error:
                code = error.diagnostics.get("http_status")
                transient = (code in (429, 500, 502, 503, 504)
                             or "failed or timed out" in str(error)
                             or "polling exceeded" in str(error))
                if not transient or attempt == 12:
                    raise
                delay = min(600, 30*2**(attempt-1))
                status(output, "retry_wait", attempt=attempt, retry_in_seconds=delay,
                       http_status=code, **common)
                print(f"Transient NVIDIA failure ({code}); resuming in {delay}s", flush=True)
                time.sleep(delay)
        status(output, "training", **common)
        torch.set_num_threads(4)
        run_phase(["train", "--output", str(output), "--device", "cpu", "--rank", "16",
                   "--batch-size", "8", "--epochs", "8", "--selection", "validation_bce",
                   "--matched-top-k-mean"])
        status(output, "complete", **common)
    except Exception as error:
        status(output, "failed", error_type=type(error).__name__, **common)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output/".full_run.lock"), timeout=0):
        if args.phase == "prepare":
            prepare(args.output)
        else:
            load_cohort(args.output)
            run(args.output)


if __name__ == "__main__":
    main()
