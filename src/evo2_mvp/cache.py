from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout

from .utils import fingerprint, load_config, save_json, sha256_file


class FeatureCache:
    """Single-writer atomic, identity-checked, resumable feature cache.

    Labels and review conclusions are deliberately excluded from feature files.
    Each manifest record is committed after its NPZ. Orphans can be regenerated.
    """
    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if metadata is not None and metadata != self.manifest["metadata"]:
                raise ValueError("Cache metadata mismatch; use a new directory for different model/FASTA/window/layer.")
        else:
            if metadata is None:
                raise FileNotFoundError(self.manifest_path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.manifest = {"schema_version": 1, "metadata": metadata, "records": {}}
            save_json(self.manifest_path, self.manifest)
        self.metadata = self.manifest["metadata"]

    def __contains__(self, variant_id):
        if variant_id not in self.manifest["records"]:
            return False
        try:
            self.get(variant_id)
            return True
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return False

    def get(self, variant_id):
        record = self.manifest["records"][variant_id]
        with np.load(self.path / record["filename"], allow_pickle=False) as archive:
            if str(archive["variant_id"].item()) != variant_id:
                raise ValueError("Cache variant identity mismatch")
            features = archive["features"].astype(np.float32)
            ref, alt, delta = (archive[k] for k in ("ref", "alt", "delta"))
            expected = np.concatenate([ref, alt, delta])
            if features.ndim != 1 or not np.isfinite(features).all() or not np.array_equal(features, expected):
                raise ValueError("Malformed/nonfinite/corrupt feature record")
            if not np.allclose(delta, alt - ref, rtol=1e-6, atol=1e-7):
                raise ValueError("Cache delta does not equal ALT - REF")
            if fingerprint(features.tolist()) != record["feature_sha256"]:
                raise ValueError("Cache feature checksum mismatch")
        return features

    def put(self, variant_id, encoded, ref_sequence, alt_sequence):
        arrays = {k: np.asarray(encoded[k], dtype=np.float32) for k in ("ref", "alt", "delta", "features")}
        ref, alt, delta, features = (arrays[k] for k in ("ref", "alt", "delta", "features"))
        if ref.ndim != 1 or ref.shape != alt.shape or ref.shape != delta.shape:
            raise ValueError("REF/ALT/delta shapes differ")
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite feature cannot be cached")
        if not np.array_equal(features, np.concatenate((ref, alt, delta))) or not np.allclose(delta, alt-ref):
            raise ValueError("Inconsistent concatenation/delta")
        if "feature_dim" in self.metadata and features.size != self.metadata["feature_dim"]:
            raise ValueError("Feature width changed")
        filename = hashlib.sha256(variant_id.encode()).hexdigest() + ".npz"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.path,
                    prefix=filename + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, variant_id=np.array(variant_id), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path / filename)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        self.manifest["records"][variant_id] = {"filename": filename,
            "feature_sha256": fingerprint(features.tolist()),
            "ref_sequence_sha256": fingerprint(ref_sequence),
            "alt_sequence_sha256": fingerprint(alt_sequence)}
        save_json(self.manifest_path, self.manifest)
        self.get(variant_id)


def _extract(args):
    from .data import load_variants
    from .sequence import ReferenceGenome
    from .evo2_encoder import create_encoder
    cfg = load_config(args.config)
    variants = load_variants(args.variants)
    if not variants:
        raise ValueError("No variants to extract")
    genome = ReferenceGenome(args.fasta)
    enc_cfg = cfg["encoder"]
    encoder = create_encoder(enc_cfg, local_checkpoint=args.local_checkpoint, download_dir=args.download_dir)
    reference_hash = sha256_file(args.fasta)
    # A complete cache resumes without making even the first remote API call.
    manifest_path = Path(args.output) / "manifest.json"
    first_index = 0
    if manifest_path.exists():
        cache = FeatureCache(args.output)
        if (cache.metadata.get("feature_identity") != encoder.metadata["feature_identity"]
                or cache.metadata.get("window") != cfg["window"]
                or cache.metadata.get("reference_sha256") != reference_hash
                or cache.metadata.get("genome_build") != cfg["genome_build"]):
            raise ValueError("Cache protocol/reference/window differs; use a new output directory")
        if all(v.id in cache for v in variants):
            print(json.dumps({"processed": 0, "skipped": len(variants), "failed": 0}))
            return
        first_index = next(i for i, variant in enumerate(variants) if variant.id not in cache)
    # First real forward establishes width and finite features before opening a new cache.
    first = genome.context(variants[first_index], cfg["window"])
    encoded = encoder.encode_pair(first.ref_sequence, first.alt_sequence, first.variant_index)
    metadata = {**encoder.metadata, "window": cfg["window"], "genome_build": cfg["genome_build"],
        "reference_sha256": reference_hash, "feature_dim": int(encoded["features"].size)}
    cache = FeatureCache(args.output, metadata)
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for i, variant in enumerate(variants):
        if variant.id in cache:
            counts["skipped"] += 1
            continue
        try:
            context = first if i == first_index else genome.context(variant, cfg["window"])
            arrays = encoded if i == first_index else encoder.encode_pair(context.ref_sequence, context.alt_sequence, context.variant_index)
            cache.put(variant.id, arrays, context.ref_sequence, context.alt_sequence)
            counts["processed"] += 1
        except Exception as exc:
            counts["failed"] += 1
            # Persist progress, but fail loudly instead of silently omitting failed variants.
            save_json(Path(args.output) / "extraction_status.json", {**counts, "failed_variant": variant.id, "error": str(exc)})
            raise
        save_json(Path(args.output) / "extraction_status.json", counts)
    print(json.dumps(counts))


def main():
    p = argparse.ArgumentParser(description="Extract real hosted Evo 2 features; one writer per cache directory.")
    p.add_argument("--variants", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--config", default="configs/mvp.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--local-checkpoint")
    p.add_argument("--download-dir", default="work/huggingface")
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Lock before loading the manifest or calling the API. Locking only save_json
    # would still allow stale in-memory manifests to overwrite each other.
    writer_lock = FileLock(str(output / ".writer.lock"), timeout=0)
    try:
        with writer_lock:
            _extract(args)
    except Timeout:
        raise RuntimeError(f"Another feature extraction process holds the writer lock for {output}. "
            "Only one extraction process may write this cache. The OS releases the lock when that process exits; "
            "do not delete the lock file while a process is running.") from None


if __name__ == "__main__":
    main()

