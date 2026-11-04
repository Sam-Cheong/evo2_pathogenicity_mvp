"""Fetch a pinned public HyenaDNA checkpoint without executing model code.

Requires requests and the Python standard library only. Every downloaded file
must match its audited SHA-256 before it can replace the destination file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REPO_ID = "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf"
REVISION = "6036d4e144922b44470294f510cbfd2539ba3b7e"
# Kept independent of torch/transformers so preparation needs only requests.
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(directory, timeout=60, max_retries=2):
    if timeout <= 0 or not 0 <= max_retries <= 5:
        raise ValueError("timeout must be positive and max_retries must be in 0..5")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    with requests.Session() as session:
        for name, expected in PINNED_FILES.items():
            destination = directory / name
            url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}"
            if destination.exists():
                if not destination.is_file() or sha256_file(destination) != expected:
                    raise ValueError(f"Existing checkpoint file failed SHA-256: {destination}. "
                                     "It cannot be used; remove or move the corrupt file explicitly before retrying.")
                status = "existing_sha256_verified"
            else:
                temporary = destination.with_name(destination.name + ".part")
                try:
                    for attempt in range(max_retries + 1):
                        try:
                            with session.get(url, stream=True, timeout=(10, timeout)) as response:
                                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                                    time.sleep(min(2 ** attempt, 5))
                                    continue
                                response.raise_for_status()
                                total = 0
                                with temporary.open("wb") as handle:
                                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                                        if not chunk:
                                            continue
                                        total += len(chunk)
                                        if total > 32 * 1024 * 1024:
                                            raise ValueError("Checkpoint file exceeds the 32MiB download bound")
                                        handle.write(chunk)
                            break
                        except requests.RequestException:
                            if attempt == max_retries:
                                raise RuntimeError(f"Public checkpoint download failed: {name}") from None
                            time.sleep(min(2 ** attempt, 5))
                    if sha256_file(temporary) != expected:
                        raise ValueError(f"Downloaded checkpoint file failed SHA-256: {name}")
                    os.replace(temporary, destination)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                status = "downloaded_sha256_verified"
            files.append({"filename": name, "source_url": url, "sha256": expected,
                          "bytes": destination.stat().st_size, "status": status})
            print(f"{name}: {status}")
    manifest = {"repo_id": REPO_ID, "revision": REVISION,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "directory": str(directory), "files": files,
                "all_pinned_files_sha256_verified": True,
                "model_card_license": "bsd-3-clause",
                "checkpoint_license_files": [],
                "license_note": "The pinned Hub repository declares BSD-3-Clause in README.md and has no standalone LICENSE file. "
                                "The upstream HazyResearch/hyena-dna code repository has a separate Apache-2.0 license; scopes are not substituted.",
                "download_operation_executes_remote_code": False,
                "model_weights_executed_by_this_script": False}
    output = directory / "download_manifest.json"
    temporary = directory / "download_manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--pretrained-dir", dest="output", default="work/hyenadna_d256_pretrained")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.output, args.timeout, args.max_retries)
    print(json.dumps({"repo_id": result["repo_id"], "revision": result["revision"],
                      "directory": result["directory"], "all_sha256_verified": True}, indent=2))


if __name__ == "__main__":
    main()
