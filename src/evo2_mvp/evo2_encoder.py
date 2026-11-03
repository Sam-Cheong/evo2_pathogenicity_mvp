from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .environment import require_official_runtime
from .utils import sha256_file

CHECKPOINT = "evo2_1b_base"
REPOSITORY = "arcinstitute/evo2_1b_base"
REVISION = "2279e1df422c991037470302360edd40d0d2ea1e"
WEIGHTS_SHA256 = "8ffba7d0e6445a8f2c92d9ff1c4e772c7f73ca9179f5e0c699b8b0ca1b966f64"


class Evo2Encoder:
    """Official Evo 2 API. No fallback model, random features or BF16 substitution.

    ``blocks.12`` is an audited existing 1B block. Its output is still validated
    at runtime; the expected 1920 hidden width is not a measured result.
    """

    def __init__(self, layer="blocks.12", pooling="variant", local_radius=16,
                 revision=REVISION, local_checkpoint=None, download_dir=None):
        require_official_runtime()
        if revision != REVISION:
            raise ValueError("This adapter is audited for revision " + REVISION)
        from huggingface_hub import hf_hub_download
        from evo2 import Evo2

        if local_checkpoint is None:
            local_checkpoint = hf_hub_download(REPOSITORY, CHECKPOINT + ".pt",
                revision=revision, cache_dir=download_dir)
        weights_sha256 = sha256_file(local_checkpoint)
        if weights_sha256 != WEIGHTS_SHA256:
            raise ValueError("Checkpoint SHA256 differs from the audited official 1B weights.")
        self.backend = Evo2(CHECKPOINT, local_path=str(local_checkpoint))
        self._initialize(layer, pooling, local_radius)
        self._metadata.update({"encoder_type": "evo2", "checkpoint": CHECKPOINT,
            "repository": REPOSITORY, "checkpoint_revision": revision,
            "source_checkpoint_sha256": weights_sha256,
            "evo2_version": importlib.metadata.version("evo2"),
            "torch_version": str(torch.__version__)})
        for distribution, field in (("vtx", "vtx_version"), ("transformer-engine", "transformer_engine_version"), ("flash-attn", "flash_attn_version")):
            self._metadata[field] = importlib.metadata.version(distribution)
        try:
            dist = importlib.metadata.distribution("evo2")
            self._metadata["evo2_install_source"] = json.loads(dist.read_text("direct_url.json") or "null")
        except (ValueError, TypeError):
            self._metadata["evo2_install_source"] = None

    def _initialize(self, layer, pooling, local_radius):
        if pooling not in ("variant", "local_mean"):
            raise ValueError("pooling must be variant or local_mean")
        if local_radius < 0:
            raise ValueError("local_radius must be nonnegative")
        self.layer, self.pooling, self.local_radius = layer, pooling, local_radius
        self.backend.model.eval()
        self.backend.model.requires_grad_(False)
        self.backend.model.get_submodule(layer)  # fail explicitly if the API or layer changed
        self.device = next(self.backend.model.parameters()).device
        self.last_token_shape = None
        self._metadata = {"encoder_type": "fake", "checkpoint": "TEST_ONLY",
            "layer": layer, "pooling": pooling, "local_radius": local_radius,
            "feature_mode": "embedding", "real_model_verified": False,
            "tokenizer": "character_level_no_bos", "orientation": "forward_reference"}

    @classmethod
    def from_backend_for_test(cls, backend, layer="blocks.0", pooling="variant", local_radius=16):
        """Explicit dependency injection used ONLY by weight-free unit tests."""
        obj = cls.__new__(cls)
        obj.backend = backend
        obj._initialize(layer, pooling, local_radius)
        return obj

    @property
    def metadata(self):
        from .provenance import feature_identity
        result = {"backend": "local", **self._metadata}
        result["feature_identity"] = feature_identity(result)
        return result

    @torch.inference_mode()
    def encode(self, sequence, variant_index):
        if not sequence or len(sequence) > 8192 or set(sequence) - set("ACGT"):
            raise ValueError("Expected 1..8192 uppercase A/C/G/T characters.")
        if not 0 <= variant_index < len(sequence):
            raise ValueError("Variant token index outside sequence.")
        token_ids = self.backend.tokenizer.tokenize(sequence)
        if len(token_ids) != len(sequence):
            raise RuntimeError("Tokenizer is not one token per nucleotide without BOS.")
        ids = torch.tensor(token_ids, dtype=torch.int, device=self.device).unsqueeze(0)
        _, embeddings = self.backend(ids, return_embeddings=True, layer_names=[self.layer])
        if self.layer not in embeddings:
            raise RuntimeError("Official API did not return the requested embedding layer.")
        hidden = embeddings[self.layer]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(sequence):
            raise RuntimeError(f"Expected (1,length,hidden) embedding, observed {tuple(hidden.shape)}")
        self.last_token_shape = list(hidden.shape)
        if self.pooling == "variant":
            pooled = hidden[0, variant_index]
        else:
            start = max(0, variant_index - self.local_radius)
            stop = min(len(sequence), variant_index + self.local_radius + 1)
            pooled = hidden[0, start:stop].float().mean(dim=0)
        result = pooled.detach().float().cpu().numpy().copy()
        if result.ndim != 1 or not np.isfinite(result).all():
            raise FloatingPointError("Invalid nonfinite embedding.")
        if self._metadata["encoder_type"] == "evo2":
            self._metadata["real_model_verified"] = True
        return result

    def encode_pair(self, ref_sequence, alt_sequence, variant_index):
        if len(ref_sequence) != len(alt_sequence):
            raise ValueError("SNV contexts must have the same length.")
        differences = [i for i, (r, a) in enumerate(zip(ref_sequence, alt_sequence)) if r != a]
        if differences != [variant_index]:
            raise ValueError("Expected exactly one REF/ALT change at variant_index.")
        ref = self.encode(ref_sequence, variant_index)
        alt = self.encode(alt_sequence, variant_index)
        delta = alt - ref
        features = np.concatenate((ref, alt, delta)).astype(np.float32)
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite REF/ALT variant features.")
        return {"ref": ref, "alt": alt, "delta": delta, "features": features}

    @torch.inference_mode()
    def score_pair(self, ref_sequence, alt_sequence):
        """Official mean sequence log-likelihood; these are NOT pathogenicity probabilities."""
        if self._metadata["encoder_type"] != "evo2":
            raise RuntimeError("Sequence scoring requires the official real model")
        scores = self.backend.score_sequences([ref_sequence, alt_sequence], batch_size=1,
            prepend_bos=False, reduce_method="mean", average_reverse_complement=False)
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise FloatingPointError("Invalid official sequence scores")
        return {"mean_log_likelihood_ref": float(scores[0]), "mean_log_likelihood_alt": float(scores[1]),
            "delta_mean_log_likelihood": float(scores[1] - scores[0]),
            "interpretation": "sequence_likelihood_only_not_pathogenicity_probability"}


def create_encoder(config, local_checkpoint=None, download_dir=None):
    """Default production backend is NVIDIA hosted API; local weights are explicit legacy opt-in."""
    cfg = dict(config)
    backend = cfg.pop("backend", "nvidia_api")
    if backend == "nvidia_api":
        from .nvidia_encoder import NvidiaEvo2Encoder
        allowed = {"model", "endpoint", "api_schema", "layer", "tensor_key", "pooling", "local_radius",
            "service_revision", "timeout_seconds", "max_retries", "poll_timeout_seconds"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError("Unknown NVIDIA encoder settings: " + ", ".join(sorted(unknown)))
        return NvidiaEvo2Encoder(**cfg)
    if backend == "local":
        return Evo2Encoder(layer=cfg["layer"], pooling=cfg["pooling"], local_radius=cfg.get("local_radius", 16),
            revision=cfg["revision"], local_checkpoint=local_checkpoint, download_dir=download_dir)
    raise ValueError("Unknown encoder backend: " + str(backend))


def encoder_config_from_metadata(metadata):
    if metadata.get("backend") == "nvidia_api":
        keys = ("model", "endpoint", "api_schema", "layer", "tensor_key", "pooling", "local_radius", "service_revision")
        return {"backend": "nvidia_api", **{key: metadata[key] for key in keys}}
    return {"backend": "local", "layer": metadata["layer"], "pooling": metadata["pooling"],
        "local_radius": metadata.get("local_radius", 16), "revision": metadata["checkpoint_revision"]}

