"""NVIDIA hosted Evo 2 forward client. No local Evo 2 weights or CUDA required."""
from __future__ import annotations

import base64
import binascii
import io
import os
import re
from pathlib import Path
import time
from urllib.parse import urlparse
import zipfile

import numpy as np
import requests

DEFAULT_MODEL = "arc/evo2-40b"
DEFAULT_ENDPOINT = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
PROFILES = {
    "nim_v1": {"layer": "blocks.20.mlp.l3", "logits_layer": "unembed"},
    "nim_v2": {"layer": "decoder.layers.20.mlp.linear_fc2", "logits_layer": "output_layer"},
}


class NvidiaAPIError(RuntimeError):
    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def safe_error_detail(response, secret):
    """Keep useful validation messages while discarding echoed input/headers."""
    allowed = {"error", "detail", "message", "msg", "loc", "type", "status", "title", "code"}

    def clean(value, depth=0):
        if depth > 5:
            return "[truncated]"
        if isinstance(value, dict):
            return {str(k): clean(v, depth+1) for k, v in value.items() if str(k).lower() in allowed}
        if isinstance(value, list):
            return [clean(v, depth+1) for v in value[:8]]
        if isinstance(value, str):
            if secret:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", value)
            value = re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", value)
            return value[:500]
        if isinstance(value, float) and not np.isfinite(value):
            return "[nonfinite]"
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return None

    try:
        payload = response.json()
    except (ValueError, TypeError):
        return None
    return clean(payload) if isinstance(payload, dict) else None


def safe_request_id(response, secret=None):
    value = response.headers.get("NVCF-REQID")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,128}", value):
        return None
    if (secret and secret in value) or "nvapi-" in value.lower():
        return None
    return value


def decode_base64_npz(value):
    """RFC-style Base64 may be line-wrapped; reject every non-whitespace invalid byte."""
    if not isinstance(value, str):
        raise ValueError("data must be a Base64 string")
    encoded = value.encode("ascii")
    encoded = re.sub(rb"[ \t\r\n\f\v]", b"", encoded)
    return base64.b64decode(encoded, validate=True)


class MissingAPIKey(NvidiaAPIError):
    pass


def load_api_key():
    from dotenv import load_dotenv
    # Explicit paths; process environment always takes precedence over .env.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    if Path.cwd() != Path(__file__).resolve().parents[2]:
        load_dotenv(Path.cwd() / ".env", override=False)
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key or key.lower() in ("your_nvidia_api_key", "replace_with_your_key", "nvapi-your-key-here"):
        raise MissingAPIKey("NVIDIA_API_KEY is missing. Put it in the project .env or the process environment; see .env.example.")
    return key


def validate_endpoint(endpoint, model):
    parsed = urlparse(endpoint)
    if model not in ("arc/evo2-40b", "arc/evo2-7b"):
        raise ValueError("Use an explicitly identified NVIDIA hosted Evo 2 40B or 7B model; hosted 1B is not assumed available.")
    if (parsed.scheme != "https" or parsed.hostname != "health.api.nvidia.com" or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != "/v1/biology/" + model + "/forward"):
        raise ValueError("NVIDIA key may only be sent to the matching official HTTPS forward endpoint.")


class NvidiaEvo2Encoder:
    def __init__(self, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT, api_schema="nim_v1",
                 layer=None, tensor_key=None, pooling="variant", local_radius=16,
                 service_revision="nvidia-hosted-run-v1", timeout_seconds=180,
                 max_retries=2, poll_timeout_seconds=180, session=None):
        from .provenance import feature_identity
        validate_endpoint(endpoint, model)
        if api_schema not in PROFILES:
            raise ValueError("api_schema must be nim_v1 or nim_v2; profiles are never silently substituted")
        layer = layer or PROFILES[api_schema]["layer"]
        prefix = "blocks." if api_schema == "nim_v1" else "decoder.layers."
        if not layer.startswith(prefix):
            raise ValueError("Select a context-dependent intermediate layer matching the explicit API schema profile")
        if pooling not in ("variant", "local_mean") or local_radius < 0:
            raise ValueError("Invalid embedding pooling configuration")
        if not service_revision or not isinstance(service_revision, str):
            raise ValueError("Set service_revision to identify this hosted experiment/cache generation")
        if timeout_seconds <= 0 or poll_timeout_seconds <= 0 or not 0 <= max_retries <= 5:
            raise ValueError("Invalid API timeout/retry settings")
        self._api_key = load_api_key()
        self.session = session or requests.Session()
        self.endpoint, self.model, self.api_schema = endpoint, model, api_schema
        self.layer, self.tensor_key = layer, tensor_key
        self.pooling, self.local_radius = pooling, local_radius
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.poll_timeout_seconds = poll_timeout_seconds
        self.last_token_shape = None
        self.last_request_info = None
        self._metadata = {"backend": "nvidia_api", "encoder_type": "evo2", "checkpoint": "nvidia/" + model,
            "model": model, "endpoint": endpoint, "api_schema": api_schema,
            "model_revision": "not_exposed_by_provider", "service_revision": service_revision,
            "layer": layer, "tensor_key": tensor_key, "pooling": pooling, "local_radius": local_radius,
            "feature_mode": "embedding", "real_model_verified": False,
            "tokenizer": "character_level_no_bos", "orientation": "forward_reference", "adapter_version": "nvidia-forward-v1"}
        self._metadata["feature_identity"] = feature_identity(self._metadata)

    @property
    def metadata(self):
        # Credentials, headers and Session state are deliberately excluded.
        return dict(self._metadata)

    def _request(self, sequence, layer):
        headers = {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json", "Accept": "application/json"}
        payload = {"sequence": sequence, "output_layers": [layer]}
        response = None
        try:
            for attempt in range(self.max_retries + 1):
                response = self.session.post(self.endpoint, headers=headers, json=payload,
                    timeout=(10, self.timeout_seconds), allow_redirects=False)
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    break
                delay = response.headers.get("Retry-After", "1")
                try:
                    delay = min(10., max(0., float(delay)))
                except ValueError:
                    delay = 1.
                time.sleep(delay)
            deadline = time.monotonic() + self.poll_timeout_seconds
            request_id = response.headers.get("NVCF-REQID")
            while response.status_code == 202:
                if not request_id or not all(c.isalnum() or c == "-" for c in request_id):
                    raise NvidiaAPIError("NVIDIA accepted an asynchronous request without a valid NVCF-REQID")
                if time.monotonic() >= deadline:
                    raise NvidiaAPIError("NVIDIA asynchronous request polling exceeded poll_timeout_seconds")
                time.sleep(1)
                response = self.session.get("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/" + request_id,
                    headers=headers, timeout=(10, min(self.timeout_seconds, max(1, deadline - time.monotonic()))), allow_redirects=False)
            # NVCF may redirect a completed large result to a presigned download.
            # The bearer credential is NEVER forwarded to the download URL.
            for _ in range(3):
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                location = response.headers.get("Location", "")
                target = urlparse(location)
                if target.scheme != "https" or not target.hostname or target.username or target.password:
                    raise NvidiaAPIError("NVIDIA result download is not a valid HTTPS location")
                response = self.session.get(location, headers={"Accept": "application/octet-stream"},
                    timeout=(10, self.timeout_seconds), allow_redirects=False)
        except requests.RequestException:
            # Never dump request objects, response bodies, headers, or exceptions containing credentials.
            raise NvidiaAPIError("NVIDIA request failed or timed out. Check network access and the configured timeout.") from None
        status = response.status_code
        if status != 200:
            reasons = {401: "API key rejected", 403: "key lacks endpoint access", 404: "forward endpoint unavailable",
                422: "request/layer schema rejected; verify nim_v1 versus nim_v2 configuration", 429: "rate/quota limit reached"}
            detail = safe_error_detail(response, self._api_key)
            hint = None
            if status == 422 and self.api_schema == "nim_v2" and "decoder" in str(detail):
                hint = ("This hosted service uses StripedHyena/Vortex layer names. Run with configs/mvp.yaml "
                    "(nim_v1, blocks.20.mlp.l3) and a separate cache/output directory; no automatic fallback was performed.")
            request_id = safe_request_id(response, self._api_key)
            diagnostics = {"http_status": status, "request_id": request_id,
                "server_detail": detail, "hint": hint}
            message = f"NVIDIA HTTP {status}: " + reasons.get(status, "remote inference failed")
            if detail:
                message += "; server detail: " + str(detail)
            if hint:
                message += "; " + hint
            raise NvidiaAPIError(message, diagnostics=diagnostics)
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type:
            result = None
            try:
                result = response.json()
                blob = decode_base64_npz(result["data"])
            except (ValueError, TypeError, KeyError, binascii.Error):
                diagnostics = {"http_status": 200, "request_id": safe_request_id(response, self._api_key),
                    "response_type": type(result).__name__,
                    "response_fields": sorted(str(k) for k in result if k in {"data", "elapsed_ms", "error", "detail", "message"}) if isinstance(result, dict) else [],
                    "data_type": type(result.get("data")).__name__ if isinstance(result, dict) else None}
                raise NvidiaAPIError("NVIDIA response is not the documented JSON data=base64(NPZ) format", diagnostics=diagnostics) from None
            elapsed = result.get("elapsed_ms")
        else:
            blob, elapsed = response.content, None
        if len(blob) > 128 * 1024 * 1024:
            raise NvidiaAPIError("NVIDIA tensor response exceeds the 128MiB client limit")
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 256 * 1024 * 1024:
                    raise NvidiaAPIError("Expanded tensor response exceeds the 256MiB client limit")
                members = archive.namelist()
                if not all(member.endswith(".npy") for member in members):
                    npz_members = [member for member in members if member.endswith(".npz")]
                    if len(npz_members) != 1:
                        raise NvidiaAPIError("Expected one NPZ result in the returned ZIP archive")
                    blob = archive.read(npz_members[0])
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 256 * 1024 * 1024:
                    raise NvidiaAPIError("Expanded NPZ tensor exceeds the 256MiB client limit")
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                candidates = [self.tensor_key] if self.tensor_key and layer == self.layer else [layer + ".output", layer]
                found = [key for key in candidates if key in archive.files]
                if len(found) != 1:
                    raise NvidiaAPIError("Requested layer tensor is missing/ambiguous; inspect the API schema and explicit tensor_key")
                array = np.asarray(archive[found[0]], dtype=np.float32)
        except (ValueError, TypeError, OSError, zipfile.BadZipFile):
            raise NvidiaAPIError("NVIDIA returned an invalid or unsupported numeric NPZ tensor") from None
        self.last_request_info = {"request_id": safe_request_id(response, self._api_key), "server_elapsed_ms": elapsed}
        return array

    @staticmethod
    def _validate_sequence(sequence):
        if not sequence or len(sequence) > 8192 or set(sequence) - set("ACGT"):
            raise ValueError("Expected 1..8192 uppercase A/C/G/T bases; this MVP keeps API windows small")

    def _normalize_shape(self, array, length):
        if array.ndim != 3:
            raise NvidiaAPIError("Expected a rank-3 layer tensor from NVIDIA")
        if array.shape[0] == 1 and array.shape[1] == length:
            normalized = array
        elif array.shape[0] == length and array.shape[1] == 1:
            normalized = array.transpose(1, 0, 2)
        else:
            raise NvidiaAPIError("Returned tensor does not align with the input sequence/token count")
        if not normalized.shape[2] or not np.isfinite(normalized).all():
            raise NvidiaAPIError("Returned tensor is empty or contains NaN/Inf")
        self.last_token_shape = list(normalized.shape)
        return normalized

    def token_activations(self, sequence):
        """Return raw activations (sequence length, hidden dimension) at the configured hook."""
        self._validate_sequence(sequence)
        return self._token_activations(sequence)

    def _token_activations(self, sequence):
        hidden = self._normalize_shape(self._request(sequence, self.layer), len(sequence))[0]
        self._metadata["real_model_verified"] = True
        return hidden

    def encode(self, sequence, variant_index):
        self._validate_sequence(sequence)
        if not 0 <= variant_index < len(sequence):
            raise ValueError("Variant token index outside sequence")
        hidden = self._token_activations(sequence)
        if self.pooling == "variant":
            result = hidden[variant_index]
        else:
            result = hidden[max(0, variant_index-self.local_radius):min(len(sequence), variant_index+self.local_radius+1)].mean(0)
        return result.astype(np.float32).copy()

    def encode_pair(self, ref_sequence, alt_sequence, variant_index):
        differences = [i for i, (r, a) in enumerate(zip(ref_sequence, alt_sequence)) if r != a]
        if len(ref_sequence) != len(alt_sequence) or differences != [variant_index]:
            raise ValueError("SNV REF/ALT must differ exactly once at variant_index")
        ref, alt = self.encode(ref_sequence, variant_index), self.encode(alt_sequence, variant_index)
        if ref.shape != alt.shape:
            raise NvidiaAPIError("REF and ALT returned different hidden dimensions; do not mix hosted deployments")
        delta = alt - ref
        features = np.concatenate((ref, alt, delta)).astype(np.float32)
        if not np.isfinite(features).all():
            raise NvidiaAPIError("Nonfinite variant features")
        return {"ref": ref, "alt": alt, "delta": delta, "features": features}

    def score_pair(self, ref_sequence, alt_sequence):
        scores = []
        for sequence in (ref_sequence, alt_sequence):
            self._validate_sequence(sequence)
            logits = self._normalize_shape(self._request(sequence, PROFILES[self.api_schema]["logits_layer"]), len(sequence))[0]
            if logits.shape[1] != 512 or len(sequence) < 2:
                raise NvidiaAPIError("Expected next-token logits (L,512) and at least two bases")
            # Causal shift: output at t predicts the input token at t+1.
            values = logits[:-1].astype(np.float64)
            maxima = values.max(axis=-1)
            log_normalizer = maxima + np.log(np.exp(values-maxima[:, None]).sum(-1))
            tokens = np.frombuffer(sequence[1:].encode("ascii"), dtype=np.uint8).astype(int)
            scores.append(float((values[np.arange(len(tokens)), tokens] - log_normalizer).mean()))
        return {"mean_log_likelihood_ref": scores[0], "mean_log_likelihood_alt": scores[1],
            "delta_mean_log_likelihood": scores[1]-scores[0], "interpretation": "sequence_likelihood_only_not_pathogenicity_probability"}

