import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evo2_mvp.cache import FeatureCache
from evo2_mvp.evo2_encoder import Evo2Encoder


class FakeBackend:
    """Deterministic toy hidden states, never presented as Evo 2 predictions."""
    def __init__(self):
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([nn.Embedding(128, 4)])
        with torch.no_grad():
            self.model.blocks[0].weight.copy_(torch.arange(512).reshape(128, 4) / 100)
        self.tokenizer = type("Tokenizer", (), {"tokenize": lambda _, seq: list(seq.encode())})()

    def __call__(self, ids, return_embeddings, layer_names):
        assert return_embeddings and layer_names == ["blocks.0"]
        return None, {"blocks.0": self.model.blocks[0](ids.long())}


def test_adapter_shape_freeze_delta_and_finite():
    backend = FakeBackend()
    encoder = Evo2Encoder.from_backend_for_test(backend)
    encoded = encoder.encode_pair("ACGT", "ATGT", 1)
    assert encoded["features"].shape == (12,)
    assert encoder.last_token_shape == [1, 4, 4]
    assert np.isfinite(encoded["features"]).all()
    np.testing.assert_array_equal(encoded["delta"], encoded["alt"] - encoded["ref"])
    assert not backend.model.training
    assert not any(p.requires_grad for p in backend.model.parameters())
    assert encoder.metadata["encoder_type"] == "fake"
    assert not encoder.metadata["real_model_verified"]
    with pytest.raises(ValueError, match="exactly one"):
        encoder.encode_pair("ACGT", "ATGA", 1)


def test_cache_identity_resume_and_metadata(tmp_path):
    encoder = Evo2Encoder.from_backend_for_test(FakeBackend())
    encoded = encoder.encode_pair("ACGT", "ATGT", 1)
    cache = FeatureCache(tmp_path, {"window": 4, "feature_dim": 12})
    cache.put("1-2-C-T", encoded, "ACGT", "ATGT")
    reloaded = FeatureCache(tmp_path)
    np.testing.assert_array_equal(reloaded.get("1-2-C-T"), encoded["features"])
    assert "1-2-C-T" in reloaded
    with pytest.raises(ValueError, match="metadata mismatch"):
        FeatureCache(tmp_path, {"window": 8})
    rec = reloaded.manifest["records"]["1-2-C-T"]
    with np.load(tmp_path / rec["filename"]) as data:
        changed = dict(data)
    changed["variant_id"] = np.array("1-3-G-T")
    np.savez(tmp_path / rec["filename"], **changed)
    with pytest.raises(ValueError, match="identity"):
        reloaded.get("1-2-C-T")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_EVO2_INTEGRATION") != "1", reason="Real NVIDIA Evo 2 API integration not run; explicit key and opt-in required")
def test_real_evo2_phase0(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/test_evo2.py"),
        "--config", str(root / "configs/mvp.yaml"), "--output", str(tmp_path)], check=True, cwd=root)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "passed" and report["real_model_verified"]
    assert report["backend"] == "nvidia_api" and report["checkpoint"] == "nvidia/arc/evo2-40b"
    assert report["finite"] and report["save_reload_consistent"]
    assert report["token_embedding_shape"][1] == 256

