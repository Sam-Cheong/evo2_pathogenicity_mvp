"""Weight-free hosted provenance guards. No network call or model claim is made."""
import copy
import json
from types import SimpleNamespace

import pytest

from evo2_mvp.distill_student import check_phase0
from evo2_mvp.provenance import feature_identity
from evo2_mvp.train_teacher_head import check_cache


@pytest.fixture
def hosted_metadata():
    # This describes a unit-test service run; it is never an inference result.
    metadata = {
        "encoder_type": "evo2", "backend": "nvidia_api", "real_model_verified": True,
        "checkpoint": "nvidia/arc/evo2-40b", "model": "arc/evo2-40b",
        "endpoint": "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward",
        "api_schema": "nim_v1", "model_revision": "unversioned_hosted",
        "service_revision": "unit-test-service-run-1", "layer": "blocks.28.mlp.l3",
        "pooling": "variant", "local_radius": 16, "feature_mode": "embedding",
        "tokenizer": "character_level_no_bos", "orientation": "forward_reference",
        "window": 256, "genome_build": "GRCh38", "reference_sha256": "a" * 64,
    }
    metadata["feature_identity"] = feature_identity(metadata)
    return metadata


def phase0_report(tmp_path, metadata, window=256, status="passed"):
    path = tmp_path / "phase0.json"
    path.write_text(json.dumps({"status": status, "real_model_verified": True,
                                "metadata": metadata, "window": window}), encoding="utf-8")
    return path


def test_hosted_cache_needs_no_invented_weight_hash(hosted_metadata):
    assert "source_checkpoint_sha256" not in hosted_metadata
    check_cache(SimpleNamespace(metadata=hosted_metadata), {"synthetic": False})


def test_hosted_cache_rejects_unverified_or_missing_run_identity(hosted_metadata):
    unverified = dict(hosted_metadata, real_model_verified=False)
    with pytest.raises(ValueError):
        check_cache(SimpleNamespace(metadata=unverified), {"synthetic": False})
    missing_revision = dict(hosted_metadata, service_revision="")
    missing_revision["feature_identity"] = feature_identity(missing_revision)
    with pytest.raises(ValueError):
        check_cache(SimpleNamespace(metadata=missing_revision), {"synthetic": False})
    missing_identity = {key: value for key, value in hosted_metadata.items() if key != "feature_identity"}
    with pytest.raises(ValueError):
        check_cache(SimpleNamespace(metadata=missing_identity), {"synthetic": False})


def test_synthetic_cache_remains_an_explicit_weight_free_bypass():
    check_cache(SimpleNamespace(metadata={"encoder_type": "fake"}), {"synthetic": True})


def test_phase0_requires_exact_hosted_encoder_identity(tmp_path, hosted_metadata):
    report = phase0_report(tmp_path, hosted_metadata)
    assert check_phase0(report, hosted_metadata)["status"] == "passed"
    other_run = dict(hosted_metadata, service_revision="unit-test-service-run-2")
    other_run["feature_identity"] = feature_identity(other_run)
    with pytest.raises(ValueError, match="identity"):
        check_phase0(report, other_run)
    other_layer = dict(hosted_metadata, layer="blocks.29.mlp.l3")
    other_layer["feature_identity"] = feature_identity(other_layer)
    with pytest.raises(ValueError, match="identity"):
        check_phase0(report, other_layer)


def test_phase0_requires_matching_window_and_passed_status(tmp_path, hosted_metadata):
    report = phase0_report(tmp_path, hosted_metadata)
    cache = dict(hosted_metadata, window=512)
    # Encoder identity excludes genomic window size; the stage gate checks both.
    cache["feature_identity"] = feature_identity(cache)
    with pytest.raises(ValueError, match="window"):
        check_phase0(report, cache)
    failed = phase0_report(tmp_path, hosted_metadata, status="failed")
    with pytest.raises(ValueError, match="not verified"):
        check_phase0(failed, hosted_metadata)
    incomplete = copy.deepcopy(hosted_metadata)
    incomplete.pop("feature_identity")
    with pytest.raises(ValueError):
        check_phase0(phase0_report(tmp_path, incomplete), hosted_metadata)
