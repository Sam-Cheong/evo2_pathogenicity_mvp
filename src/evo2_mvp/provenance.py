"""Feature protocol identity for local and hosted Evo 2 experiments."""
from .utils import fingerprint

REMOTE_IDENTITY_FIELDS = ("backend", "encoder_type", "checkpoint", "model", "endpoint", "api_schema",
    "model_revision", "service_revision", "layer", "tensor_key", "pooling", "local_radius",
    "feature_mode", "tokenizer", "orientation", "adapter_version")
LOCAL_IDENTITY_FIELDS = ("backend", "encoder_type", "checkpoint", "checkpoint_revision", "source_checkpoint_sha256",
    "layer", "pooling", "local_radius", "feature_mode", "tokenizer", "orientation", "evo2_version",
    "vtx_version", "transformer_engine_version", "flash_attn_version", "torch_version", "evo2_install_source")


def feature_identity(metadata):
    fields = REMOTE_IDENTITY_FIELDS if metadata.get("backend") == "nvidia_api" else LOCAL_IDENTITY_FIELDS
    return fingerprint({key: metadata.get(key) for key in fields})


def validate_evo2_metadata(metadata, require_verified=True):
    if metadata.get("encoder_type") != "evo2":
        raise ValueError("Feature cache must identify a real Evo 2 encoder")
    if require_verified and metadata.get("real_model_verified") is not True:
        raise ValueError("Real Evo 2 inference has not been verified")
    if metadata.get("feature_mode") != "embedding":
        raise ValueError("This experiment requires embedding features, not an unlabelled scoring baseline")
    if metadata.get("backend") == "nvidia_api":
        from .nvidia_encoder import validate_endpoint
        validate_endpoint(metadata.get("endpoint", ""), metadata.get("model", ""))
        if metadata.get("checkpoint") != "nvidia/" + metadata["model"]:
            raise ValueError("Hosted model identity is inconsistent")
        for key in ("service_revision", "model_revision", "layer", "pooling", "feature_identity"):
            if not metadata.get(key):
                raise ValueError("Hosted feature metadata is missing " + key)
        if metadata.get("api_schema") not in ("nim_v1", "nim_v2"):
            raise ValueError("Unknown NVIDIA API schema profile")
        prefix = "blocks." if metadata["api_schema"] == "nim_v1" else "decoder.layers."
        if not metadata["layer"].startswith(prefix):
            raise ValueError("Layer naming does not match the explicit NVIDIA API schema profile")
    else:
        if metadata.get("checkpoint") != "evo2_1b_base" or not metadata.get("checkpoint_revision"):
            raise ValueError("Unrecognized legacy local Evo 2 checkpoint")
        if len(metadata.get("source_checkpoint_sha256", "")) != 64:
            raise ValueError("Missing local checkpoint SHA256")
    if metadata.get("feature_identity") != feature_identity(metadata):
        raise ValueError("Feature identity checksum mismatch")

