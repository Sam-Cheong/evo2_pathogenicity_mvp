"""Label-isolated train-only Evo 2 representation data for two-stage students.

Stage 1 receives no labels: its inner split uses connected components of shared
genes and overlapping reference-coordinate windows. Clinical validation/test
Teacher feature files are never opened by this module.
"""
from __future__ import annotations

from bisect import bisect_left
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np

from .cache import FeatureCache
from .data import genes_of, load_variants, verify_splits
from .provenance import validate_evo2_metadata
from .sequence import ReferenceGenome, Variant
from .utils import fingerprint, save_json, sha256_file

TEACHER_WIDTH = 8192
REPRESENTATION_KEYS = {"ref", "alt", "teacher_ref", "teacher_alt", "variant_ids"}
PROBE_KEYS = {"ref", "alt", "labels", "variant_ids"}
OUTPUT_NAMES = ("representation_train.npz", "representation_val.npz",
                "probe_train.npz", "probe_val.npz", "probe_test.npz")


def _read_unlabelled_variants(path):
    variants, sources = [], set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"chrom", "pos", "ref", "alt", "gene"} <= set(reader.fieldnames or []):
            raise ValueError("Variant CSV needs chrom,pos,ref,alt,gene")
        for row in reader:
            source = str(row.get("source", row.get("data_source", "unspecified"))).strip()
            if source.lower() == "synthetic" or any(str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}
                                                      for key in ("synthetic", "is_synthetic")):
                raise ValueError("Two-stage research data must contain no synthetic records")
            if row.get("genome_build", "GRCh38") != "GRCh38":
                raise ValueError("Only GRCh38 variants are accepted")
            # Labels are deliberately not parsed or copied at this stage.
            variants.append(Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"], None))
            sources.add(source)
    if not variants or len({v.id for v in variants}) != len(variants):
        raise ValueError("Variant CSV must be nonempty and contain no duplicate SNVs")
    return variants, sorted(sources)


def context_interval(variant, window):
    start = variant.pos - 1 - window // 2
    return variant.chrom, start, start + window


def _interval_lookup(ids, lookup, window):
    result = {}
    for variant_id in ids:
        chrom, start, end = context_interval(lookup[variant_id], window)
        result.setdefault(chrom, []).append((start, end, variant_id))
    indexed = {}
    for chrom, intervals in result.items():
        intervals.sort()
        maximum, maxima = -1, []
        for _, end, _ in intervals:
            maximum = max(maximum, end)
            maxima.append(maximum)
        indexed[chrom] = ([x[0] for x in intervals], intervals, maxima)
    return indexed


def _overlaps(variant, window, indexed):
    chrom, start, end = context_interval(variant, window)
    if chrom not in indexed:
        return []
    starts, intervals, maxima = indexed[chrom]
    i, hits = bisect_left(starts, end) - 1, []
    while i >= 0 and maxima[i] > start:
        left, right, variant_id = intervals[i]
        if left < end and right > start:
            hits.append(variant_id)
        i -= 1
    return sorted(hits)


def representation_partition(variants, splits, window=256, seed=42):
    """Return an inner train/val partition without inspecting variant.label."""
    if window != 256:
        raise ValueError("This two-stage protocol requires exactly 256-bp contexts")
    verify_splits(variants, splits)
    lookup = {v.id: v for v in variants}
    held_out = _interval_lookup(splits["val"] + splits["test"], lookup, window)
    excluded = {key: hits for key in splits["train"]
                if (hits := _overlaps(lookup[key], window, held_out))}
    retained = [key for key in splits["train"] if key not in excluded]
    if len(retained) < 2:
        raise ValueError("Fewer than two outer-train records remain after clinical-context overlap exclusion")
    parent = {key: key for key in retained}

    def find(key):
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            previous = parent[key]
            parent[key] = root
            key = previous
        return root

    def union(left, right):
        parent[find(right)] = find(left)

    gene_owner = {}
    for key in sorted(retained):
        genes = genes_of(lookup[key])
        if not genes:
            raise ValueError("Reliable representation gene grouping requires known genes: " + key)
        for gene in sorted(genes):
            if gene in gene_owner:
                union(key, gene_owner[gene])
            else:
                gene_owner[gene] = key
    # Connected interval components, including transitive overlaps, are joined.
    by_chrom = _interval_lookup(retained, lookup, window)
    for _, intervals, _ in by_chrom.values():
        anchor, component_end = None, -1
        for start, end, key in intervals:
            if anchor is not None and start < component_end:
                union(anchor, key)
                component_end = max(component_end, end)
            else:
                anchor, component_end = key, end
    components = {}
    for key in sorted(retained):
        components.setdefault(find(key), []).append(key)
    groups = sorted(components.values(), key=lambda group: tuple(group))
    if len(groups) < 2:
        raise ValueError("Only one connected gene/context group remains; inner representation validation would leak")
    random.Random(seed).shuffle(groups)
    # Subset sum finds the feasible validation size closest to 20%, independent
    # of class labels. Seed affects group tie choices, never label information.
    possible = {0: ()}
    for index, group in enumerate(groups):
        for size, chosen in list(possible.items()):
            candidate_size = size + len(group)
            if candidate_size < len(retained) and candidate_size not in possible:
                possible[candidate_size] = chosen + (index,)
    target = 0.2 * len(retained)
    chosen_size = min((size for size in possible if size > 0), key=lambda size: (abs(size - target), size))
    val_set = {key for index in possible[chosen_size] for key in groups[index]}
    train_ids = [key for key in retained if key not in val_set]
    val_ids = [key for key in retained if key in val_set]
    train_genes = set().union(*(genes_of(lookup[key]) for key in train_ids))
    val_genes = set().union(*(genes_of(lookup[key]) for key in val_ids))
    if train_genes & val_genes:
        raise AssertionError("Representation gene leakage")
    val_intervals = _interval_lookup(val_ids, lookup, window)
    if any(_overlaps(lookup[key], window, val_intervals) for key in train_ids):
        raise AssertionError("Representation context leakage")
    if any(_overlaps(lookup[key], window, held_out) for key in retained):
        raise AssertionError("Representation overlaps clinical held-out context")
    return {"train": train_ids, "val": val_ids, "excluded": excluded,
            "retained": retained, "connected_group_count": len(groups),
            "validation_fraction": len(val_ids) / len(retained),
            "validation_target_fraction": 0.2,
            "algorithm": "seeded_gene_and_context_connected_components_nearest_size_subset_sum_no_labels"}


def _one_hot(contexts, attribute, window):
    mapping = {ord(base): index for index, base in enumerate("ACGT")}
    result = np.zeros((len(contexts), 4, window), dtype=np.float32)
    positions = np.arange(window)
    for index, context in enumerate(contexts):
        sequence = getattr(context, attribute)
        channels = np.fromiter((mapping[base] for base in sequence.encode("ascii")), dtype=np.int64)
        result[index, channels, positions] = 1.0
    return result


def _atomic_npz(path, arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_output_archive(path, expected_ids, representation, window=256):
    expected_keys = REPRESENTATION_KEYS if representation else PROBE_KEYS
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("Unexpected or forbidden NPZ keys: " + str(path))
        if archive["variant_ids"].dtype.kind not in ("U", "S") or archive["variant_ids"].tolist() != expected_ids:
            raise ValueError("NPZ variant order/identity mismatch")
        n = len(expected_ids)
        for name in ("ref", "alt"):
            value = archive[name]
            if value.shape != (n, 4, window) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid DNA one-hot array")
            if not np.isin(value, [0., 1.]).all() or not (value.sum(axis=1) == 1).all():
                raise ValueError("DNA input is not strict one-hot")
        if representation:
            for name in ("teacher_ref", "teacher_alt"):
                value = archive[name]
                if value.shape != (n, TEACHER_WIDTH) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Expected real cached Teacher REF/ALT arrays (N,8192) float32")
        else:
            labels = archive["labels"]
            if labels.shape != (n,) or labels.dtype != np.float32 or not np.isin(labels, [0., 1.]).all():
                raise ValueError("Probe labels must be a finite binary vector")


def prepare_two_stage(variants_path, splits_path, cache_path, fasta_path,
                      output="outputs/sh2_two_stage_20261104/data", seed=42, window=256):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if window != 256:
        raise ValueError("The requested SH2 protocol requires window=256")
    variants_path, splits_path, cache_path, fasta_path = map(Path, (variants_path, splits_path, cache_path, fasta_path))
    output = Path(output)
    variants, sources = _read_unlabelled_variants(variants_path)
    raw_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = raw_splits.get("splits", raw_splits)
    partition = representation_partition(variants, splits, window, seed)
    cache = FeatureCache(cache_path)
    validate_evo2_metadata(cache.metadata, require_verified=True)
    if cache.metadata.get("genome_build") != "GRCh38" or cache.metadata.get("window") != window:
        raise ValueError("Cache GRCh38/window identity mismatch")
    if cache.metadata.get("feature_dim") != 3 * TEACHER_WIDTH:
        raise ValueError("The source Teacher must expose actual 8192-dimensional REF/ALT embeddings")
    fasta_sha = sha256_file(fasta_path)
    if cache.metadata.get("reference_sha256") != fasta_sha:
        raise ValueError("FASTA SHA256 differs from the Teacher feature cache")
    source_hashes = {"variants_sha256": sha256_file(variants_path), "splits_sha256": sha256_file(splits_path),
                     "fasta_sha256": fasta_sha, "cache_manifest_sha256": sha256_file(cache.manifest_path)}
    preparation_identity = fingerprint({"source_hashes": source_hashes, "cache_metadata": cache.metadata,
                                        "window": window, "seed": seed, "schema_version": 1})
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("status") != "complete" or existing.get("preparation_identity") != preparation_identity:
            raise ValueError("Existing two-stage output is incomplete or belongs to different inputs; use a new output directory")
        for name in OUTPUT_NAMES:
            path = output / name
            if not path.is_file() or sha256_file(path) != existing["files_sha256"][name]:
                raise ValueError("Existing two-stage output checksum mismatch: " + name)
            ids = existing["stage1_train_ids" if name == "representation_train.npz" else "stage1_val_ids"] if name.startswith("representation_") else splits[name.removeprefix("probe_").removesuffix(".npz")]
            validate_output_archive(path, ids, name.startswith("representation_"), window)
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("Two-stage output contains partial artifacts without completed metadata; use a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    lookup = {v.id: v for v in variants}
    with ReferenceGenome(fasta_path) as genome:
        contexts = {key: genome.context(lookup[key], window) for name in ("train", "val", "test") for key in splits[name]}
    teacher_arrays, teacher_file_hashes, teacher_feature_hashes = {}, {}, {}
    outer_train = set(splits["train"])
    teacher_ids_read = []
    for variant_id in partition["retained"]:
        if variant_id not in outer_train:
            raise AssertionError("A held-out Teacher NPZ was requested")
        record = cache.manifest["records"].get(variant_id)
        if record is None:
            raise ValueError("Missing train-only Teacher cache record: " + variant_id)
        expected_filename = hashlib.sha256(variant_id.encode()).hexdigest() + ".npz"
        if record.get("filename") != expected_filename:
            raise ValueError("Unexpected Teacher cache filename for train variant")
        context = contexts[variant_id]
        if (record.get("ref_sequence_sha256") != fingerprint(context.ref_sequence)
                or record.get("alt_sequence_sha256") != fingerprint(context.alt_sequence)):
            raise ValueError("Teacher cache REF/ALT context hash mismatch: " + variant_id)
        teacher_ids_read.append(variant_id)
        verified_features = cache.get(variant_id)  # Existing FeatureCache SHA/delta checks.
        if verified_features.shape != (3 * TEACHER_WIDTH,):
            raise ValueError("Expected cached Teacher feature shape (24576,)")
        teacher_path = cache.path / expected_filename
        with np.load(teacher_path, allow_pickle=False) as archive:
            reference = np.asarray(archive["ref"], dtype=np.float32)
            alternate = np.asarray(archive["alt"], dtype=np.float32)
        if reference.shape != (TEACHER_WIDTH,) or alternate.shape != (TEACHER_WIDTH,):
            raise ValueError("Cached Teacher REF and ALT must each have actual shape (8192,)")
        if not np.array_equal(reference, verified_features[:TEACHER_WIDTH]) or not np.array_equal(alternate, verified_features[TEACHER_WIDTH:2*TEACHER_WIDTH]):
            raise ValueError("Teacher cache changed between verified feature and REF/ALT reads")
        teacher_arrays[variant_id] = (reference, alternate)
        teacher_file_hashes[variant_id] = sha256_file(teacher_path)
        teacher_feature_hashes[variant_id] = record["feature_sha256"]
    files_sha256 = {}
    for name, ids in (("train", partition["train"]), ("val", partition["val"])):
        selected_contexts = [contexts[key] for key in ids]
        arrays = {"ref": _one_hot(selected_contexts, "ref_sequence", window),
                  "alt": _one_hot(selected_contexts, "alt_sequence", window),
                  "teacher_ref": np.stack([teacher_arrays[key][0] for key in ids]).astype(np.float32),
                  "teacher_alt": np.stack([teacher_arrays[key][1] for key in ids]).astype(np.float32),
                  "variant_ids": np.asarray(ids, dtype=str)}
        path = output / f"representation_{name}.npz"
        _atomic_npz(path, arrays)
        validate_output_archive(path, ids, True, window)
        files_sha256[path.name] = sha256_file(path)

    # Only after the label-independent representation IDs have been finalized
    # are the CSV labels parsed. They enter probe archives exclusively.
    labelled = load_variants(variants_path)
    labelled_by_id = {v.id: v for v in labelled}
    if any(v.label not in (0, 1) for v in labelled):
        raise ValueError("Every probe record needs a genuine binary ClinVar label")
    data_provenance = {"variants_sha256": source_hashes["variants_sha256"],
        "variant_label_fingerprint": fingerprint(sorted((v.id, v.label, v.gene) for v in labelled)),
        "n_variants": len(labelled), "synthetic": False, "sources": sources, "performance_scope": "research_dataset"}
    for name in ("train", "val", "test"):
        ids = splits[name]
        selected_contexts = [contexts[key] for key in ids]
        arrays = {"ref": _one_hot(selected_contexts, "ref_sequence", window),
                  "alt": _one_hot(selected_contexts, "alt_sequence", window),
                  "labels": np.asarray([labelled_by_id[key].label for key in ids], dtype=np.float32),
                  "variant_ids": np.asarray(ids, dtype=str)}
        path = output / f"probe_{name}.npz"
        _atomic_npz(path, arrays)
        validate_output_archive(path, ids, False, window)
        files_sha256[path.name] = sha256_file(path)
    if sha256_file(cache.manifest_path) != source_hashes["cache_manifest_sha256"]:
        raise RuntimeError("Source Teacher cache manifest changed during preparation")
    if (sha256_file(variants_path) != source_hashes["variants_sha256"]
            or sha256_file(splits_path) != source_hashes["splits_sha256"]):
        raise RuntimeError("Source variants or clinical splits changed during preparation")
    counts = {"outer_train": len(splits["train"]), "outer_val": len(splits["val"]), "outer_test": len(splits["test"]),
              "excluded_train_context_overlap": len(partition["excluded"]),
              "stage1_retained_train_only": len(partition["retained"]),
              "representation_train": len(partition["train"]), "representation_val": len(partition["val"]),
              "probe_train": len(splits["train"]), "probe_val": len(splits["val"]), "probe_test": len(splits["test"]),
              "connected_representation_groups": partition["connected_group_count"]}
    metadata = {"schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_identity": preparation_identity, "random_seed": seed, "window": window, "genome_build": "GRCh38",
        "synthetic": False, "data_provenance": data_provenance, "cache_metadata": cache.metadata,
        "split_fingerprint": fingerprint(splits), "source_hashes": source_hashes,
        "source_csv": str(variants_path.resolve()), "source_splits": str(splits_path.resolve()),
        "source_cache": str(cache_path.resolve()), "source_fasta": str(fasta_path.resolve()),
        "files_sha256": files_sha256, "counts": counts,
        "stage1_train_ids": partition["train"], "stage1_val_ids": partition["val"],
        "representation_train_ids": partition["train"], "representation_val_ids": partition["val"],
        "stage1_train_ids_sha256": fingerprint(partition["train"]), "stage1_val_ids_sha256": fingerprint(partition["val"]),
        "representation_train_ids_sha256": fingerprint(partition["train"]), "representation_val_ids_sha256": fingerprint(partition["val"]),
        "probe_ids": {name: list(splits[name]) for name in ("train", "val", "test")},
        "split_ids": {name: list(splits[name]) for name in ("train", "val", "test")},
        "outer_train_ids": list(splits["train"]), "outer_val_ids": list(splits["val"]), "outer_test_ids": list(splits["test"]),
        "excluded_train_ids": sorted(partition["excluded"]), "excluded_train_overlapping_clinical_ids": partition["excluded"],
        "teacher_feature_ids_read": teacher_ids_read, "source_teacher_npz_sha256": teacher_file_hashes,
        "source_teacher_feature_sha256": teacher_feature_hashes,
        "representation_partition": {key: partition[key] for key in ("algorithm", "validation_fraction", "validation_target_fraction", "connected_group_count")},
        "validation": {"representation_split_does_not_read_labels": True, "stage1_npz_contains_no_labels_genes_reviews": True,
                       "stage1_gene_disjoint": True, "stage1_context_disjoint": True,
                       "stage1_has_no_clinical_val_test_context_overlap": True,
                       "clinical_teacher_npz_never_opened": True, "original_probe_split_unchanged": True,
                       "feature_cache_checksums_verified": True, "teacher_ref_alt_context_hashes_verified": True,
                       "teacher_ref_alt_width": TEACHER_WIDTH, "all_outputs_finite": True},
        "clinical_use": "research_only", "labels_location": "probe_train.npz/probe_val.npz/probe_test.npz only; metadata retains legacy one-way provenance fingerprint, no label values"}
    save_json(metadata_path, metadata)
    return metadata
