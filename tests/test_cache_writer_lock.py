"""Process-level cache writer and atomic-file tests; no API or model inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock, Timeout
import numpy as np
import pytest

from evo2_mvp import cache as cache_module
from evo2_mvp import utils


PROJECT = Path(__file__).resolve().parents[1]


def child_environment():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT / "src")
    environment["NVIDIA_API_KEY"] = "unit-test-no-provider-access"
    environment["RUN_EVO2_INTEGRATION"] = "0"
    return environment


def run_child(code, *arguments):
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, arguments)],
        cwd=PROJECT, env=child_environment(), capture_output=True,
        text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


ACQUIRE_ONCE = """
import sys
from filelock import FileLock
with FileLock(sys.argv[1], timeout=0):
    print("acquired", flush=True)
"""


def test_second_cli_rejected_before_input_read_or_api_and_release_allows_next_writer(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    lock_path = output / ".writer.lock"
    with FileLock(str(lock_path), timeout=0):
        # None of the inputs exists: the lock refusal must occur before any
        # config/data read and therefore before construction of an API encoder.
        result = subprocess.run(
            [sys.executable, "-m", "evo2_mvp.cache",
             "--variants", str(tmp_path / "missing-variants.csv"),
             "--fasta", str(tmp_path / "missing-reference.fa"),
             "--config", str(tmp_path / "missing-config.yaml"),
             "--output", str(output)],
            cwd=PROJECT, env=child_environment(), capture_output=True,
            text=True, timeout=45,
        )
        diagnostic = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Another feature extraction process holds the writer lock" in diagnostic
        assert "Only one extraction process may write this cache" in diagnostic
        assert "FileNotFoundError" not in diagnostic
        assert not (output / "manifest.json").exists()
        assert not (output / "extraction_status.json").exists()
        assert not list(output.glob("*.npz"))
    assert "acquired" in run_child(ACQUIRE_ONCE, lock_path).stdout


def test_cli_exception_releases_writer_lock(tmp_path, monkeypatch):
    output = tmp_path / "cache"
    lock_path = output / ".writer.lock"
    entered = []

    def failing_extract(args):
        entered.append(True)
        # The production main must hold the actual lock throughout _extract.
        with pytest.raises(Timeout):
            with FileLock(str(lock_path), timeout=0):
                pass
        raise PermissionError("intentional test-only extraction exception")

    monkeypatch.setattr(cache_module, "_extract", failing_extract)
    monkeypatch.setattr(sys, "argv", [
        "cache", "--variants", "unused.csv", "--fasta", "unused.fa",
        "--config", "unused.yaml", "--output", str(output),
    ])
    with pytest.raises(PermissionError, match="intentional test-only"):
        cache_module.main()
    assert entered == [True]
    assert "acquired" in run_child(ACQUIRE_ONCE, lock_path).stdout


def test_os_releases_lock_after_writer_child_is_terminated(tmp_path):
    lock_path = tmp_path / ".writer.lock"
    code = """
import sys
from filelock import FileLock
with FileLock(sys.argv[1], timeout=0):
    print("locked", flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        cwd=PROJECT, env=child_environment(), text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(Timeout):
            with FileLock(str(lock_path), timeout=0):
                pass
        process.terminate()
        process.wait(timeout=15)
        # Do not delete the lock file to recover: closing the terminated
        # process's OS handles must make the same path available again.
        assert "acquired" in run_child(ACQUIRE_ONCE, lock_path).stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()


def test_json_replace_failure_preserves_old_and_only_cleans_own_unique_temporary(tmp_path, monkeypatch):
    target = tmp_path / "extraction_status.json"
    utils.save_json(target, {"processed": 7})
    another_writer_temp = tmp_path / "extraction_status.json.another-writer.tmp"
    another_writer_temp.write_text("owned by another writer", encoding="utf-8")
    attempted_sources = []

    def reject_replace(source, destination):
        source, destination = Path(source), Path(destination)
        attempted_sources.append(source)
        assert destination == target
        assert source.parent == target.parent
        assert source != target.with_suffix(".json.tmp")
        assert source != another_writer_temp
        assert source.exists()
        assert json.loads(source.read_text(encoding="utf-8")) == {"processed": 8}
        raise PermissionError("intentional test-only replace failure")

    monkeypatch.setattr(utils.os, "replace", reject_replace)
    for _ in range(2):
        with pytest.raises(PermissionError, match="intentional test-only"):
            utils.save_json(target, {"processed": 8})
        assert json.loads(target.read_text(encoding="utf-8")) == {"processed": 7}
        assert another_writer_temp.read_text(encoding="utf-8") == "owned by another writer"
    assert len(set(attempted_sources)) == 2
    assert all(not source.exists() for source in attempted_sources)
    assert list(tmp_path.glob("*.tmp")) == [another_writer_temp]


def test_serial_writer_processes_reload_manifest_and_preserve_prior_variant(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    code = """
import sys
from pathlib import Path
from filelock import FileLock
import numpy as np
from evo2_mvp.cache import FeatureCache
path, variant_id = Path(sys.argv[1]), sys.argv[2]
with FileLock(str(path / ".writer.lock"), timeout=0):
    # A fresh cache object is opened only after acquiring the writer lock.
    # Reusing an object loaded before locking would retain a stale manifest.
    cache = FeatureCache(path, {"feature_dim": 6, "window": 4, "encoder_type": "fake"})
    ref = np.array([1., 2.], dtype=np.float32)
    alt = np.array([2., 3.], dtype=np.float32)
    delta = alt - ref
    cache.put(variant_id, {"ref": ref, "alt": alt, "delta": delta,
                          "features": np.concatenate([ref, alt, delta])}, "ACGT", "ATGT")
    print(variant_id, flush=True)
"""
    ids = ["1-2-C-T", "1-3-G-A"]
    for variant_id in ids:
        run_child(code, output, variant_id)
    cache = cache_module.FeatureCache(output)
    assert set(cache.manifest["records"]) == set(ids)
    for variant_id in ids:
        np.testing.assert_array_equal(cache.get(variant_id), [1., 2., 2., 3., 1., 1.])
    assert len(list(output.glob("*.npz"))) == 2
    assert not list(output.glob("*.tmp"))
