"""Local input checks and clinical gene groups for the EVEE pilot."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .data import _gene_groups, genes_of
from .metrics import file_sha256, fingerprint
from .sequence import Variant
from .two_stage_student import load_encoder_pair, load_metadata, load_partition, model_kwargs


@dataclass(frozen=True)
class PilotInputs:
    train: dict
    val: dict
    encoder_artifacts: dict
    provenance: dict


def _verify_source(data_dir, variants_csv, splits_path=None):
    metadata, metadata_sha = load_metadata(data_dir)
    provenance = metadata.get("data_provenance", {})
    if metadata.get("status") != "complete" or provenance.get("synthetic") is not False:
        raise ValueError("The EVEE pilot requires the completed real-data SH2 preparation")
    if metadata.get("synthetic") is not False:
        raise ValueError("Synthetic inputs are outside the EVEE pilot")
    variants_csv = Path(variants_csv)
    splits_path = Path(splits_path) if splits_path is not None else variants_csv.with_name("splits.json")
    expected = metadata["source_hashes"]
    variants_sha, splits_sha = file_sha256(variants_csv), file_sha256(splits_path)
    if variants_sha != expected["variants_sha256"] or variants_sha != provenance["variants_sha256"]:
        raise ValueError("Original variant CSV checksum changed")
    if splits_sha != expected["splits_sha256"]:
        raise ValueError("Original clinical split checksum changed")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = splits.get("splits", splits)
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("Clinical split must contain train, val and test IDs")
    all_ids = []
    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids or any(not isinstance(key, str) or not key for key in ids):
            raise ValueError("Invalid clinical split IDs: " + name)
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("The original clinical split has duplicate or overlapping IDs")
    if (metadata.get("split_ids") != splits
            or metadata.get("split_fingerprint") != fingerprint(splits)
            or provenance.get("n_variants") != len(all_ids)):
        raise ValueError("Prepared clinical IDs differ from the original split")
    return metadata, metadata_sha, splits, variants_sha, splits_sha


def _gene_groups_for_splits(variants_csv, splits, names):
    """Restore connected gene groups using the selected IDs and gene symbols only."""
    selected = {key for name in names for key in splits[name]}
    annotations = {}
    with Path(variants_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"chrom", "pos", "ref", "alt", "gene"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Original variant CSV lacks clinical gene annotations")
        for row in reader:
            key = row.get("variant_id")
            if not key:
                key = Variant(row["chrom"], row["pos"], row["ref"], row["alt"]).id
            if key not in selected:
                continue
            variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"])
            if variant.id != key or key in annotations:
                raise ValueError("Variant annotation identity is inconsistent: " + key)
            annotations[key] = variant
    if set(annotations) != selected:
        raise ValueError("Missing clinical gene annotations for the locked variants")
    group_by_id = {}
    for component in _gene_groups([annotations[key] for key in sorted(annotations)]):
        # A stable component label retains all connected genes. A multi-gene
        # variant therefore groups every connected record in CV and bootstrap.
        genes = sorted(set().union(*(genes_of(variant) for variant in component)))
        group = "|".join(genes)
        for variant in component:
            group_by_id[variant.id] = group
    result = {name: np.asarray([group_by_id[key] for key in splits[name]], dtype=str) for name in names}
    seen = set()
    for name, groups in result.items():
        current = set(groups.tolist())
        if seen & current:
            raise ValueError("Connected clinical gene groups overlap the outer split: " + name)
        seen.update(current)
    return result


def _partition(data_dir, metadata, splits, split, groups):
    partition = load_partition(data_dir, f"probe_{split}.npz", metadata, representation=False)
    if partition["variant_ids"] != splits[split]:
        raise ValueError("Probe order or variant IDs differ from the original clinical split: " + split)
    partition["groups"] = groups
    return partition


def load_pilot_inputs(data_dir, encoders_dir, variants_csv, encoder_config, *, splits_path=None):
    """Validate and load the two development splits and the locked encoder pair."""
    metadata, metadata_sha, splits, variants_sha, splits_sha = _verify_source(data_dir, variants_csv, splits_path)
    initial, distilled = load_encoder_pair(encoders_dir, metadata_sha, encoder_config)
    artifacts = {"random": initial, "distilled": distilled}
    for name, artifact in artifacts.items():
        if artifact.get("model_kwargs") != model_kwargs(encoder_config, "linear"):
            raise ValueError("Source encoder model configuration differs: " + name)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in artifact["encoder_state"].values()):
            raise ValueError("Nonfinite tensor in source encoder: " + name)
    encoder_sha256s = {name: value["encoder_sha256"] for name, value in artifacts.items()}
    if encoder_sha256s["random"] == encoder_sha256s["distilled"]:
        raise ValueError("Distillation did not change the source encoder")
    groups = _gene_groups_for_splits(variants_csv, splits, ("train", "val"))
    train = _partition(data_dir, metadata, splits, "train", groups["train"])
    val = _partition(data_dir, metadata, splits, "val", groups["val"])
    provenance = {
        "prepared_metadata_sha256": metadata_sha,
        "split_fingerprint": metadata["split_fingerprint"],
        "variants_sha256": variants_sha,
        "splits_sha256": splits_sha,
        "encoder_config_sha256": fingerprint(encoder_config),
        "encoder_sha256s": encoder_sha256s,
        "encoder_file_sha256s": {
            "random": file_sha256(Path(encoders_dir) / "initial_encoder.pt"),
            "distilled": file_sha256(Path(encoders_dir) / "best_encoder.pt"),
        },
        "source_data_provenance": metadata["data_provenance"],
        "cache_metadata": metadata["cache_metadata"],
        "development_probe_sha256s": {
            name: metadata["files_sha256"][f"probe_{name}.npz"] for name in ("train", "val")
        },
        "development_group_counts": {
            name: int(np.unique(groups[name]).size) for name in ("train", "val")
        },
        "development_groups_sha256": {
            name: fingerprint(list(zip(splits[name], groups[name].tolist()))) for name in ("train", "val")
        },
    }
    return PilotInputs(train, val, artifacts, provenance)


def load_historical_test(data_dir, variants_csv, lock, *, splits_path=None):
    """Load the historical holdout only after the caller validates its decision lock.

    The evaluation entry point validates the checkpoint files against the lock.
    This boundary verifies the lock status and binds access to the prepared data.
    """
    if not isinstance(lock, dict) or lock.get("status") != "locked_for_historical_test":
        raise ValueError("Historical test requires a locked validation decision")
    checkpoint_hashes = lock.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in checkpoint_hashes.values()
    ):
        raise ValueError("Historical test lock is missing verified checkpoint hashes")
    metadata, metadata_sha, splits, _, _ = _verify_source(data_dir, variants_csv, splits_path)
    if lock.get("prepared_metadata_sha256") != metadata_sha:
        raise ValueError("Historical test lock belongs to different prepared data")
    groups = _gene_groups_for_splits(variants_csv, splits, ("train", "val", "test"))
    return _partition(data_dir, metadata, splits, "test", groups["test"])


def grouped_folds(partition, n_splits=5, seed=42):
    """Return reproducible group-disjoint, class-stratified OOF index pairs."""
    labels = np.asarray(partition["labels"])
    groups = np.asarray(partition["groups"])
    if labels.ndim != 1 or groups.shape != labels.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("Grouped folds require aligned one-dimensional groups and binary labels")
    if (not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2
            or np.unique(groups).size < n_splits):
        raise ValueError("Grouped folds require at least n_splits independent clinical gene groups")
    if any(np.unique(groups[labels == label]).size < n_splits for label in (0, 1)):
        raise ValueError("Too few independent groups containing each class for the requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    held_out = np.concatenate([validation for _, validation in folds])
    if not np.array_equal(np.sort(held_out), np.arange(len(labels))):
        raise ValueError("Grouped folds do not hold out each variant exactly once")
    for fit, validation in folds:
        if np.intersect1d(groups[fit], groups[validation]).size:
            raise ValueError("A clinical gene group crosses grouped CV boundaries")
        if np.unique(labels[fit]).size != 2 or np.unique(labels[validation]).size != 2:
            raise ValueError("A grouped CV fold lacks one class")
    return folds
