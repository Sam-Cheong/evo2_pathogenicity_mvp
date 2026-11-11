"""Short-window, bidirectional Evo 2 7B features for public ClinVar SNVs."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
from threading import Lock, local
import zipfile

from filelock import FileLock, Timeout
import numpy as np

from .data import genes_of
from .nvidia_encoder import NvidiaAPIError, validate_endpoint
from .sequence import ReferenceContext, ReferenceGenome, Variant
from .two_stage_data import _atomic_npz
from .utils import fingerprint, save_json, sha256_file


ORIENTATIONS = ("forward", "reverse_complement")
FEATURE_DIM = 4096
METHOD = "evee_style_short_window_v1"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_ADAPTER_FIELDS = (
    "model", "endpoint", "api_schema", "model_revision", "service_revision",
    "layer", "tensor_key", "tokenizer", "adapter_version",
)
_RECORD_ERRORS = (KeyError, ValueError, OSError, EOFError, zipfile.BadZipFile)


def _variant_index(window):
    if type(window) is not int or not 3 <= window <= 8192:
        raise ValueError("Expected an integer window between 3 and 8192 bases")
    return 3 * window // 4


def variant_contexts(genome, variant, window):
    """Get a strand-relative 3:1 window ending with the variant and its downstream context."""
    index = _variant_index(window)
    forward = genome.context(variant, window, variant_index=index)
    backward = genome.context(variant, window, variant_index=window - 1 - index)
    reverse = ReferenceContext(
        backward.ref_sequence.translate(_COMPLEMENT)[::-1],
        backward.alt_sequence.translate(_COMPLEMENT)[::-1],
        index,
    )
    return forward, reverse


def _sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _direction_arrays(ref, alt, variant_index, top_k):
    """Select the most cosine-divergent paired positions, breaking ties by position."""
    ref, alt = ref[variant_index:], alt[variant_index:]
    delta = alt - ref
    ref_norm, alt_norm = np.linalg.norm(ref, axis=1), np.linalg.norm(alt, axis=1)
    if any(not np.isfinite(norm).all() or (norm <= 0).any() for norm in (ref_norm, alt_norm)):
        raise NvidiaAPIError("Cannot compute cosine divergence for zero or nonfinite activation norms")
    # For unit vectors, 0.5 * squared distance equals 1 - cosine similarity.
    # This avoids subtracting nearly equal floats when two activations are close.
    difference = alt / alt_norm[:, None] - ref / ref_norm[:, None]
    divergence = 0.5 * np.einsum("ij,ij->i", difference, difference)
    selected = np.sort(np.argsort(-divergence, kind="stable")[:top_k])
    return {
        "delta": delta[selected],
        "mean": delta.mean(axis=0, dtype=np.float32),
        "positions": (selected + variant_index).astype(np.int64),
    }


def _metadata(encoder, window, top_k, reference_sha256):
    index = _variant_index(window)
    if type(top_k) is not int or not 1 <= top_k <= window - index:
        raise ValueError("top_k must be between 1 and the number of variant/downstream positions")
    if not isinstance(reference_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_sha256):
        raise ValueError("Expected a verified SHA-256 digest for the GRCh38 FASTA")
    encoder_metadata = encoder.metadata
    remote = {key: encoder_metadata.get(key) for key in _ADAPTER_FIELDS}
    if remote["model"] != "arc/evo2-7b":
        raise ValueError("The EVEE feature workflow requires the NVIDIA Evo 2 7B forward endpoint")
    validate_endpoint(remote["endpoint"], remote["model"])
    layer_prefix = {"nim_v1": "blocks.27", "nim_v2": "decoder.layers.27"}.get(remote["api_schema"])
    layer = remote["layer"]
    if layer_prefix is None or not isinstance(layer, str) or not (
        layer == layer_prefix or layer.startswith(layer_prefix + ".")
    ):
        raise ValueError("Choose an explicit Evo 2 7B block-27 hook for the selected API schema")
    if remote["tokenizer"] != "character_level_no_bos":
        raise ValueError("Expected one aligned activation per base without special tokens")
    protocol = {
        **remote,
        "method": METHOD,
        "genome_build": "GRCh38",
        "reference_sha256": reference_sha256,
        "window": window,
        "top_k": top_k,
        "variant_index": index,
        "feature_dim": FEATURE_DIM,
        "orientations": list(ORIENTATIONS),
        "position_semantics": "zero_based_orientation_token_offset",
        "effect_region": "variant_and_downstream",
        "selection": "top_cosine_divergence_stable_ties_sorted_by_position",
        "delta": "alt_minus_ref_float32",
        "mean": "full_effect_region_delta_float32",
    }
    return {**protocol, "feature_identity": fingerprint(protocol)}


class EVEEFeatureCache:
    """Committed per-direction arrays for one verified feature extraction protocol."""

    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        if not self.manifest_path.exists():
            if metadata is None:
                raise FileNotFoundError(self.manifest_path)
            save_json(self.manifest_path, {"schema_version": 1, "metadata": metadata, "records": {}})
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.metadata = self.manifest["metadata"]
        protocol = {key: value for key, value in self.metadata.items() if key != "feature_identity"}
        if (
            self.manifest.get("schema_version") != 1
            or self.metadata.get("method") != METHOD
            or self.metadata.get("feature_identity") != fingerprint(protocol)
            or not isinstance(self.manifest.get("records"), dict)
        ):
            raise ValueError("Invalid EVEE feature cache manifest")
        if metadata is not None and metadata != self.metadata:
            raise ValueError("EVEE feature cache identity changed; use a new cache directory")

    def _path(self, variant_id, orientation):
        return self.path / f"{fingerprint(variant_id)}.{orientation}.npz"

    def _validate(self, arrays):
        delta, mean, positions = (arrays[key] for key in ("delta", "mean", "positions"))
        k, width = self.metadata["top_k"], self.metadata["feature_dim"]
        if (
            delta.dtype != np.float32 or delta.shape != (k, width)
            or mean.dtype != np.float32 or mean.shape != (width,)
            or not np.isfinite(delta).all() or not np.isfinite(mean).all()
            or positions.dtype != np.int64 or positions.shape != (k,)
            or np.any(positions < self.metadata["variant_index"])
            or np.any(positions >= self.metadata["window"])
            or np.any(np.diff(positions) <= 0)
        ):
            raise ValueError("Invalid EVEE delta, mean, or positions")

    def _read(self, variant_id, orientation):
        record = self.manifest["records"][variant_id][orientation]
        path = self._path(variant_id, orientation)
        if sha256_file(path) != record["sha256"]:
            raise ValueError("EVEE feature record checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"variant_id", "orientation", "delta", "mean", "positions"}:
                raise ValueError("Unexpected EVEE feature record fields")
            if archive["variant_id"].item() != variant_id or archive["orientation"].item() != orientation:
                raise ValueError("EVEE feature record identity mismatch")
            arrays = {key: archive[key] for key in ("delta", "mean", "positions")}
        self._validate(arrays)
        return arrays

    def get(self, variant_id):
        """Return forward/reverse stacks with shapes [2,K,D], [2,D], and [2,K]."""
        forward, reverse = (self._read(variant_id, direction) for direction in ORIENTATIONS)
        return {key: np.stack((forward[key], reverse[key])) for key in forward}

    def _put(self, variant_id, orientation, arrays, context, request_ids):
        self._validate(arrays)
        path = self._path(variant_id, orientation)
        _atomic_npz(path, {**arrays, "variant_id": np.array(variant_id), "orientation": np.array(orientation)})
        self.manifest["records"].setdefault(variant_id, {})[orientation] = {
            "sha256": sha256_file(path),
            "ref_sequence_sha256": _sequence_hash(context.ref_sequence),
            "alt_sequence_sha256": _sequence_hash(context.alt_sequence),
            "request_ids": request_ids,
        }
        save_json(self.manifest_path, self.manifest)


def _fetch_direction(encoder, context, top_k):
    states, request_ids = [], []
    for sequence in (context.ref_sequence, context.alt_sequence):
        hidden = np.asarray(encoder.token_activations(sequence), dtype=np.float32)
        if hidden.shape != (len(sequence), FEATURE_DIM) or not np.isfinite(hidden).all():
            raise NvidiaAPIError("Expected finite, base-aligned Evo 2 7B activations with width 4096")
        states.append(hidden)
        request_ids.append((encoder.last_request_info or {}).get("request_id"))
    return _direction_arrays(*states, context.variant_index, top_k), request_ids


def _missing_directions(cache, variant_id, contexts):
    missing = []
    for orientation, context in zip(ORIENTATIONS, contexts):
        try:
            cache._read(variant_id, orientation)
        except _RECORD_ERRORS:
            missing.append((orientation, context))
    return missing


def _extract_parallel(cache, variants, genome, window, top_k, workers, encoder_factory, report):
    thread = local()
    encoders = []
    encoder_lock = Lock()

    def fetch(context):
        if not hasattr(thread, "encoder"):
            encoder = encoder_factory()
            if _metadata(encoder, window, top_k, cache.metadata["reference_sha256"]) != cache.metadata:
                raise ValueError("Parallel encoder identity differs from the cache")
            session = getattr(encoder, "session", None)
            with encoder_lock:
                if any(
                    other is encoder or (session is not None and getattr(other, "session", None) is session)
                    for other in encoders
                ):
                    raise ValueError("Each parallel worker requires a distinct encoder and HTTP session")
                encoders.append(encoder)
                close = getattr(session, "close", None)
                if callable(close):
                    sessions.callback(close)
            thread.encoder = encoder
        return _fetch_direction(thread.encoder, context, top_k)

    pending = {}
    remaining = {}
    failure = None

    def record_failure(variant_id, exc):
        nonlocal failure
        if failure is None:
            failure = (variant_id, exc)
            for future in pending:
                future.cancel()

    def collect():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            variant_id, orientation, context = pending.pop(future)
            if future.cancelled():
                continue
            try:
                arrays, request_ids = future.result()
                cache._put(variant_id, orientation, arrays, context, request_ids)
                remaining[variant_id] -= 1
                if remaining[variant_id] == 0:
                    del remaining[variant_id]
                    report(variant_id, "processed")
            except Exception as exc:
                record_failure(variant_id, exc)

    with ExitStack() as sessions, ThreadPoolExecutor(max_workers=workers) as pool:
        for variant in variants:
            if failure is not None:
                break
            try:
                contexts = variant_contexts(genome, variant, window)
                missing = _missing_directions(cache, variant.id, contexts)
                if not missing:
                    report(variant.id, "skipped")
                    continue
                remaining[variant.id] = len(missing)
                for orientation, context in missing:
                    if len(pending) == workers or any(future.done() for future in pending):
                        collect()
                    if failure is not None:
                        break
                    future = pool.submit(fetch, context)
                    pending[future] = (variant.id, orientation, context)
            except Exception as exc:
                record_failure(variant.id, exc)
        # Save completed in-flight directions even when another request has failed.
        while pending:
            collect()
    if failure is not None:
        variant_id, exc = failure
        report(variant_id, "failed")
        raise exc


def extract_features(encoder, variants, fasta_path, output, *, window=256, top_k=32,
                     reference_sha256=None, progress=None, workers=1, encoder_factory=None):
    """Cache features for caller-verified public ClinVar SNVs and resume committed directions.

    Supply reference_sha256 only when the caller has already hashed the FASTA.
    The optional progress callback receives the variant ID and counts after each record.
    Parallel workers each use a separate encoder from the zero-argument encoder_factory.
    """
    if type(workers) is not int or workers < 1 or (workers > 1 and not callable(encoder_factory)):
        raise ValueError("workers must be a positive integer with an encoder_factory for parallel extraction")
    variants = list(variants)
    if not variants or any(
        not isinstance(variant, Variant) or variant.label not in (0, 1) or not genes_of(variant)
        for variant in variants
    ):
        raise ValueError("Expected public ClinVar autosomal SNVs with binary labels and gene annotations")
    if len({variant.id for variant in variants}) != len(variants):
        raise ValueError("Duplicate variant ID")
    reference_sha256 = sha256_file(fasta_path) if reference_sha256 is None else reference_sha256
    metadata = _metadata(encoder, window, top_k, reference_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"processed": 0, "skipped": 0, "failed": 0}

    def report(variant_id, state):
        counts[state] += 1
        status = {**counts, "failed_variant": variant_id} if state == "failed" else counts
        save_json(output / "extraction_status.json", status)
        if progress is not None and state != "failed":
            progress(variant_id, counts.copy())

    try:
        with FileLock(str(output / ".writer.lock"), timeout=0), ReferenceGenome(fasta_path) as genome:
            cache = EVEEFeatureCache(output, metadata)
            if workers > 1:
                _extract_parallel(cache, variants, genome, window, top_k, workers, encoder_factory, report)
                return counts
            for variant in variants:
                try:
                    contexts = variant_contexts(genome, variant, window)
                    missing = _missing_directions(cache, variant.id, contexts)
                    for orientation, context in missing:
                        arrays, request_ids = _fetch_direction(encoder, context, top_k)
                        cache._put(variant.id, orientation, arrays, context, request_ids)
                except Exception:
                    report(variant.id, "failed")
                    raise
                report(variant.id, "processed" if missing else "skipped")
    except Timeout:
        raise RuntimeError(f"Another extraction process is writing EVEE features in {output}") from None
    return counts
