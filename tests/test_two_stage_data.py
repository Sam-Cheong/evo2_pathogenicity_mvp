"""Synthetic array fixtures verify isolation; these tests report no model performance."""
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from evo2_mvp.cache import FeatureCache
from evo2_mvp.provenance import feature_identity
from evo2_mvp.sequence import ReferenceGenome, Variant
from evo2_mvp.two_stage_data import (
    PROBE_KEYS, REPRESENTATION_KEYS, context_interval, prepare_two_stage,
    representation_partition,
)
from evo2_mvp.utils import fingerprint, save_json, sha256_file


@pytest.fixture
def fixture_data(tmp_path):
    fasta = tmp_path / "GRCh38_fixture.fa"
    fasta.write_text(">1\n" + "ACGT" * 3000 + "\n", encoding="ascii")
    positions = [400, 700, 1000, 1200, 1600, 1900, 2300, 2600, 1500, 3500]
    genes = ["A", "A", "B", "C", "D", "E", "F", "G", "VAL", "TEST"]
    values = [Variant("1", pos, "ACGT"[(pos-1) % 4], "ACGT"[pos % 4], gene, i % 2)
              for i, (pos, gene) in enumerate(zip(positions, genes))]
    splits = {"train": [v.id for v in values[:8]], "val": [values[8].id], "test": [values[9].id]}
    csv_path, split_path = tmp_path / "variants.csv", tmp_path / "splits.json"
    def write_variants(path, flipped=False):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["chrom", "pos", "ref", "alt", "gene", "label", "source", "synthetic"])
            writer.writeheader()
            for value in values:
                writer.writerow({"chrom": value.chrom, "pos": value.pos, "ref": value.ref, "alt": value.alt,
                    "gene": value.gene, "label": 1-value.label if flipped else value.label,
                    "source": "ClinVar", "synthetic": "false"})
    write_variants(csv_path)
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    metadata = {"backend": "nvidia_api", "encoder_type": "evo2", "checkpoint": "nvidia/arc/evo2-40b",
        "model": "arc/evo2-40b", "endpoint": "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward",
        "api_schema": "nim_v1", "model_revision": "not_exposed_by_provider", "service_revision": "unit_fixture_only",
        "layer": "blocks.20.mlp.l3", "tensor_key": None, "pooling": "variant", "local_radius": 16,
        "feature_mode": "embedding", "real_model_verified": True, "tokenizer": "character_level_no_bos",
        "orientation": "forward_reference", "adapter_version": "nvidia-forward-v1", "genome_build": "GRCh38",
        "reference_sha256": sha256_file(fasta), "window": 256, "feature_dim": 24576}
    metadata["feature_identity"] = feature_identity(metadata)
    cache_path = tmp_path / "cache"
    cache = FeatureCache(cache_path, metadata)
    # Deliberately no validation/test NPZ exists: touching them must fail.
    with ReferenceGenome(fasta) as genome:
        for i, value in enumerate(values[:8]):
            reference = np.full(8192, i / 10, dtype=np.float32)
            alternate = reference + np.float32(.01)
            delta = alternate-reference
            context = genome.context(value, 256)
            cache.put(value.id, {"ref": reference, "alt": alternate, "delta": delta,
                                 "features": np.concatenate((reference, alternate, delta))},
                      context.ref_sequence, context.alt_sequence)
    return {"csv": csv_path, "splits_path": split_path, "splits": splits,
            "cache": cache_path, "fasta": fasta, "variants": values,
            "write_variants": write_variants, "tmp": tmp_path}


def run_fixture(data, name="prepared"):
    return prepare_two_stage(data["csv"], data["splits_path"], data["cache"], data["fasta"],
                             data["tmp"] / name, seed=42, window=256)


def test_representation_split_is_label_independent_and_component_safe(fixture_data):
    data = fixture_data
    a = representation_partition(data["variants"], data["splits"])
    b = representation_partition([replace(v, label=1-v.label) for v in data["variants"]], data["splits"])
    assert a == b
    assert set(a["excluded"]) == {data["variants"][4].id}
    assigned = {key: name for name in ("train", "val") for key in a[name]}
    assert assigned[data["variants"][0].id] == assigned[data["variants"][1].id]  # same gene
    assert assigned[data["variants"][2].id] == assigned[data["variants"][3].id]  # overlapping 256bp
    assert set(a["train"]) | set(a["val"]) == set(data["splits"]["train"]) - set(a["excluded"])
    assert not set(a["train"]) & set(a["val"])


def test_prepare_schema_no_label_or_gene_in_representation_and_train_only_teacher_reads(fixture_data, monkeypatch):
    data, calls = fixture_data, []
    original = FeatureCache.get
    def audited_get(self, variant_id):
        assert variant_id in data["splits"]["train"]
        calls.append(variant_id)
        return original(self, variant_id)
    monkeypatch.setattr(FeatureCache, "get", audited_get)
    before_splits = data["splits_path"].read_bytes()
    result = run_fixture(data)
    assert result["status"] == "complete"
    assert result["counts"]["probe_train"] == 8
    assert result["counts"]["probe_val"] == result["counts"]["probe_test"] == 1
    assert result["counts"]["stage1_retained_train_only"] == 7
    assert set(calls) == set(result["stage1_train_ids"]) | set(result["stage1_val_ids"])
    assert data["splits_path"].read_bytes() == before_splits
    for split in ("train", "val"):
        with np.load(data["tmp"] / "prepared" / f"representation_{split}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == REPRESENTATION_KEYS
            assert archive["ref"].shape[1:] == (4, 256)
            assert archive["teacher_ref"].shape[1:] == (8192,)
            assert archive["teacher_ref"].dtype == np.float32
            assert not {"label", "labels", "gene", "review"} & set(archive.files)
    for split in ("train", "val", "test"):
        with np.load(data["tmp"] / "prepared" / f"probe_{split}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == PROBE_KEYS
            assert archive["variant_ids"].tolist() == data["splits"][split]
    assert result["cache_metadata"] == FeatureCache(data["cache"]).metadata
    assert result["split_fingerprint"] == fingerprint(data["splits"])
    assert result["data_provenance"]["synthetic"] is False
    assert all(result["validation"][name] for name in ("stage1_gene_disjoint", "stage1_context_disjoint", "clinical_teacher_npz_never_opened"))


def test_label_changes_affect_only_probe_labels_and_provenance(fixture_data):
    data = fixture_data
    first = run_fixture(data, "first")
    data["write_variants"](data["csv"], flipped=True)
    second = run_fixture(data, "second")
    assert first["stage1_train_ids"] == second["stage1_train_ids"]
    assert first["stage1_val_ids"] == second["stage1_val_ids"]
    assert first["data_provenance"]["variant_label_fingerprint"] != second["data_provenance"]["variant_label_fingerprint"]
    for split in ("train", "val"):
        with np.load(data["tmp"] / "first" / f"representation_{split}.npz", allow_pickle=False) as a, np.load(data["tmp"] / "second" / f"representation_{split}.npz", allow_pickle=False) as b:
            assert all(np.array_equal(a[key], b[key]) for key in REPRESENTATION_KEYS)


def test_context_hash_mismatch_rejected(fixture_data):
    data = fixture_data
    cache = FeatureCache(data["cache"])
    cache.manifest["records"][data["splits"]["train"][0]]["ref_sequence_sha256"] = "0" * 64
    save_json(cache.manifest_path, cache.manifest)
    with pytest.raises(ValueError, match="context hash mismatch"):
        run_fixture(data)


def test_feature_cache_sha_mismatch_rejected(fixture_data):
    data = fixture_data
    cache = FeatureCache(data["cache"])
    cache.manifest["records"][data["splits"]["train"][0]]["feature_sha256"] = "0" * 64
    save_json(cache.manifest_path, cache.manifest)
    with pytest.raises(ValueError, match="feature checksum mismatch"):
        run_fixture(data)


def test_existing_outputs_reused_only_after_checksums_and_config_match(fixture_data):
    data = fixture_data
    first = run_fixture(data)
    assert run_fixture(data) == first
    target = data["tmp"] / "prepared" / "representation_train.npz"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="output checksum mismatch"):
        run_fixture(data)


def test_wrong_fasta_sha_rejected(fixture_data):
    data = fixture_data
    data["fasta"].write_text(">1\n" + "TGCA" * 3000 + "\n")
    with pytest.raises(ValueError, match="FASTA SHA256"):
        run_fixture(data)


def test_wrong_teacher_width_rejected(fixture_data):
    data = fixture_data
    cache = FeatureCache(data["cache"])
    cache.manifest["metadata"]["feature_dim"] = 768
    save_json(cache.manifest_path, cache.manifest)
    with pytest.raises(ValueError, match="8192-dimensional"):
        run_fixture(data)


def test_unsupported_window_and_single_connected_group_fail(fixture_data):
    data = fixture_data
    with pytest.raises(ValueError, match="256"):
        representation_partition(data["variants"], data["splits"], window=512)
    values = [replace(v, gene="SAME_TRAIN") if v.id in data["splits"]["train"] else v for v in data["variants"]]
    with pytest.raises(ValueError, match="one connected"):
        representation_partition(values, data["splits"])


def test_context_interval_uses_half_open_fasta_coordinates():
    variant = Variant("1", 401, "A", "G", "G")
    assert context_interval(variant, 256) == ("1", 272, 528)
