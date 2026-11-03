"""End-to-end CLI wiring with mocked NVIDIA tensors, never real inference evidence."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from evo2_mvp.evo2_encoder import create_encoder
from evo2_mvp.nvidia_encoder import NvidiaEvo2Encoder


class MockSession:
    def __init__(self):
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        sequence = kwargs["json"]["sequence"]
        layer = kwargs["json"]["output_layers"][0]
        values = np.array([[ord(base), i, 1.] for i, base in enumerate(sequence)], dtype=np.float32)[None]
        stream = io.BytesIO()
        np.savez(stream, **{layer + ".output": values})
        payload = {"data": base64.b64encode(stream.getvalue()).decode(), "elapsed_ms": 1}
        return type("Response", (), {"status_code": 200, "headers": {"Content-Type": "application/json"},
            "json": lambda _: payload})()


def test_phase0_api_writes_reloadable_arrays_without_cuda(tmp_path, monkeypatch):
    import dotenv
    import torch
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("NVIDIA_API_KEY", "unit-test-placeholder-never-sent-to-provider")
    session = MockSession()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("api_phase0_script", root / "scripts/test_evo2.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "create_encoder", lambda cfg, **kw: NvidiaEvo2Encoder(session=session, **{k: v for k, v in cfg.items() if k != "backend"}))
    monkeypatch.setattr(script, "inspect_environment", lambda **kw: {"selected_backend_blockers": []})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("API path must not synchronize CUDA")))
    out = tmp_path / "phase0_mock_only"
    monkeypatch.setattr(sys, "argv", ["test_evo2.py", "--config", str(root / "configs/mvp.yaml"), "--output", str(out)])
    assert script.main() == 0
    report = json.loads((out / "result.json").read_text())
    assert report["backend"] == "nvidia_api" and report["embedding_shape"] == [3]
    assert report["gpu_peak_allocated_bytes"] is None
    assert report["variant_features_shape"] == [9]
    assert np.load(out / "features.npy").shape == (9,)
    assert report["save_reload_consistent"] and session.calls == 4


def test_api_cache_resume_makes_no_remote_requests(tmp_path, monkeypatch):
    import dotenv
    from evo2_mvp import cache, evo2_encoder
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("NVIDIA_API_KEY", "unit-test-placeholder-never-sent-to-provider")
    session = MockSession()
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(evo2_encoder, "create_encoder", lambda cfg, **kw: NvidiaEvo2Encoder(session=session, **{k: v for k, v in cfg.items() if k != "backend"}))
    fasta = tmp_path / "fixture.fa"
    fasta.write_text(">1\n" + "ACGT" * 300 + "\n")
    variants = tmp_path / "fixture.csv"
    variants.write_text("chrom,pos,ref,alt,gene,label,source\n1,401,A,G,UNIT,1,synthetic\n")
    output = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["cache", "--variants", str(variants), "--fasta", str(fasta),
        "--config", str(root / "configs/mvp.yaml"), "--output", str(output)])
    cache.main()
    calls = session.calls
    assert calls == 2
    cache.main()
    assert session.calls == calls
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metadata"]["backend"] == "nvidia_api"
    assert "NVIDIA_API_KEY" not in json.dumps(manifest)

