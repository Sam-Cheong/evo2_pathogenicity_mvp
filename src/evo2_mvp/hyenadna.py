"""Verified pretrained HyenaDNA d256 with causal variant-center features.

Only initialization reads local pretrained files. A trained student checkpoint
contains the entire backbone and is independently reloadable without the Hub,
NVIDIA credentials, the teacher, or the original pretrained directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Transformers 4.x may inspect/migrate its cache on import. Keep that cache
# inside the active project, including when the user's home is read-only.
os.environ.setdefault("HF_HOME", str(Path.cwd() / "work" / "huggingface"))

import torch
from torch import nn

from .vendor.hyenadna.configuration_hyena import HyenaConfig
from .vendor.hyenadna.modeling_hyena import HyenaDNAModel

PRETRAINED_ID = "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf"
PRETRAINED_REVISION = "6036d4e144922b44470294f510cbfd2539ba3b7e"
PINNED_FILES = {
    "model.safetensors": "81c15c4042d5fae646f00b429b76c65644cdbe88aa7e98b61aa781c84ff09031",
    "config.json": "2b3209dc6d176d0cf573e0cdadbedb4ab0c5f7456136474200c5d827a126d336",
    "configuration_hyena.py": "29cb6cae67f5010e279d67349e688663043f0bd6791a09c36d96da8cac3668a6",
    "modeling_hyena.py": "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607",
    "tokenization_hyena.py": "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408",
    "tokenizer_config.json": "b1a98c1d2f4e1a5582feffeaccfe46e160f2f8e927009b49d4676dd79e7a9694",
    "special_tokens_map.json": "026831cec9f2ac5e501aa9d3a8b3a93c021aebe55813574c493e40b49ad28582",
    "README.md": "18ee2eeab45dd561dfd266e8756d267389cef724839e4250f093acfb7c9c57b8",
}
TOKEN_POLICY = {
    "base_to_id": {"A": 7, "C": 8, "G": 9, "T": 10},
    "add_special_tokens": False, "padding": False, "truncation": False,
    "representation": "last_hidden_state_at_variant_index",
    "orientation": "forward_reference_causal", "variant_index": "window//2",
}


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    """Stable tensor-content fingerprint, independent of torch.save ZIP metadata."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class HyenaDNAStudent(nn.Module):
    def __init__(self, encoder_config: dict, head_dims=(256, 64), dropout=0.2,
                 pooling="variant_center"):
        super().__init__()
        if pooling != "variant_center":
            raise ValueError("This experiment requires variant_center pooling")
        config = HyenaConfig(**encoder_config)
        if config.d_model != 256:
            raise ValueError("hyenadna_d256 requires a 256-dimensional backbone")
        if len(head_dims) != 2 or any(int(x) <= 0 for x in head_dims):
            raise ValueError("Expected two positive head dimensions")
        self.encoder = HyenaDNAModel(config)
        self.feature_dim = 3 * config.d_model
        self.max_length = min(1024, int(config.max_seq_len))
        self.head = nn.Sequential(nn.Linear(self.feature_dim, int(head_dims[0])), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(int(head_dims[0]), int(head_dims[1])),
                                  nn.GELU(), nn.Linear(int(head_dims[1]), 1))
        if sum(p.numel() for p in self.parameters()) >= 2_000_000:
            raise ValueError("Student encoder plus head must contain fewer than 2M parameters")

    def token_ids(self, sequence):
        """Convert the project's strict A/C/G/T one-hot input to verified IDs."""
        if sequence.ndim != 3 or sequence.shape[1] != 4 or sequence.shape[0] < 1:
            raise ValueError("Expected nonempty student input (B, 4, L)")
        if not 1 <= sequence.shape[-1] <= self.max_length:
            raise ValueError(f"DNA length must be in 1..{self.max_length}; no silent truncation")
        if (not torch.isfinite(sequence).all() or not ((sequence == 0) | (sequence == 1)).all()
                or not (sequence.sum(dim=1) == 1).all()):
            raise ValueError("Expected strict finite A/C/G/T one-hot input")
        return sequence.argmax(dim=1).long() + 7

    def encode(self, sequence, variant_index=None):
        tokens = self.token_ids(sequence)
        center = tokens.shape[1] // 2 if variant_index is None else variant_index
        if not isinstance(center, int) or isinstance(center, bool) or not 0 <= center < tokens.shape[1]:
            raise ValueError("variant_index must address a DNA base, not a special token")
        hidden = self.encoder(tokens, output_hidden_states=False, return_dict=True).last_hidden_state
        return hidden[:, center, :]

    def forward(self, reference, alternate, variant_index=None):
        if reference.shape != alternate.shape:
            raise ValueError("REF and ALT must have the same batch/context dimensions")
        ref = self.encode(reference, variant_index)
        alt = self.encode(alternate, variant_index)
        features = torch.cat([ref, alt, alt - ref], dim=-1)
        return self.head(features).squeeze(-1), features


def load_pretrained_encoder(encoder, directory):
    """Load the entire verified backbone, restoring only proven shared aliases."""
    from safetensors.torch import load_file

    path = Path(directory) / "model.safetensors"
    if file_digest(path) != PINNED_FILES["model.safetensors"]:
        raise ValueError("Pretrained weight SHA256 differs from the pinned official checkpoint")
    saved = load_file(str(path), device="cpu")
    if set(k for k in saved if not k.startswith("hyena.")) != {"lm_head.weight"}:
        raise ValueError("Unexpected non-backbone tensors in pretrained checkpoint")
    state = {k.removeprefix("hyena."): value for k, value in saved.items() if k.startswith("hyena.")}
    parameters = dict(encoder.named_parameters(remove_duplicate=False))
    aliases = {}
    for key in encoder.state_dict():
        if key in state:
            continue
        # Each layer reuses the exact same HyenaSin Parameter in its filter MLP.
        # Safetensors stores that tensor once. No unrelated missing key is allowed.
        candidates = [other for other in state if key in parameters and other in parameters
                      and parameters[key] is parameters[other]]
        if not candidates or not key.endswith(".freq"):
            raise ValueError(f"Missing pretrained backbone tensor: {key}")
        aliases[key] = candidates[0]
        state[key] = state[candidates[0]]
    encoder.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in encoder.state_dict().values()):
        raise ValueError("Pretrained backbone contains NaN/Inf")
    return {"strict_load": True, "restored_shared_aliases": aliases,
            "discarded_pretrained_tensors": ["lm_head.weight"],
            "encoder_initial_state_sha256": state_digest(encoder.state_dict())}


def initialize_pretrained_student(student_config, window):
    directory = Path(student_config.get("pretrained_dir", "work/hyenadna_d256_pretrained"))
    bundled = Path(__file__).parent / "vendor" / "hyenadna"
    bundled_hashes = {}
    for name in ("configuration_hyena.py", "modeling_hyena.py", "tokenization_hyena.py"):
        bundled_hashes[name] = file_digest(bundled / name)
        if bundled_hashes[name] != PINNED_FILES[name]:
            raise ValueError(f"Bundled official source checksum mismatch: {name}")
    for filename, expected in PINNED_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/prepare_hyenadna.py first")
        if file_digest(path) != expected:
            raise ValueError(f"Pinned pretrained file checksum mismatch: {filename}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    kwargs = {"encoder_config": config, "head_dims": list(student_config.get("head_dims", [256, 64])),
              "dropout": float(student_config.get("dropout", 0.2)),
              "pooling": student_config.get("pooling", "variant_center")}
    model = HyenaDNAStudent(**kwargs)
    if not isinstance(window, int) or not 3 <= window <= model.max_length:
        raise ValueError(f"Window must be in 3..{model.max_length}")
    loading = load_pretrained_encoder(model.encoder, directory)
    metadata = {"repo_id": PRETRAINED_ID, "revision": PRETRAINED_REVISION,
                "files_sha256": dict(PINNED_FILES), "bundled_sources_sha256": bundled_hashes,
                "token_policy": TOKEN_POLICY,
                "max_length": model.max_length, "pretrained_encoder_loaded": True,
                "encoder_finetuned": True, "parameter_count": sum(p.numel() for p in model.parameters()),
                "encoder_parameter_count": sum(p.numel() for p in model.encoder.parameters()),
                "model_card_license": "bsd-3-clause", **loading}
    return model, kwargs, metadata
