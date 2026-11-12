from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # Close the file before replacement: Windows cannot replace an open temp file.
        for delay in (0.025, 0.05, 0.1, 0.2, 0.4, None):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Windows readers can briefly block replacement of the destination.
                if delay is None or getattr(exc, "winerror", None) not in (5, 32, 33):
                    raise
                time.sleep(delay)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Preserve any original write/replace exception.


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

