"""Mock-only HTTP diagnostic tests; fixtures are not hosted inference evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import requests

from evo2_mvp.nvidia_encoder import NvidiaAPIError, NvidiaEvo2Encoder


FAKE_KEY = "unit-test-private-token-with-no-provider-access"
PROVIDER_ERROR = "'StripedHyena' object has no attribute 'decoder'"


class Response:
    def __init__(self, payload=None, *, status=422, content_type="application/json", request_id="test-request-422"):
        self.status_code = status
        self.headers = {"Content-Type": content_type, "NVCF-REQID": request_id}
        self.payload = payload
        self.content = ("unstructured echoed secret " + FAKE_KEY).encode()
        self.json_calls = 0

    @property
    def text(self):
        raise AssertionError("Unstructured provider text must not enter diagnostics")

    def json(self):
        self.json_calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert len(self.calls) == 1, "422 must not silently retry a different layer/model/schema"
        return self.response

    def get(self, *args, **kwargs):
        raise AssertionError("These 422 fixtures need no polling or fallback request")


@pytest.fixture(autouse=True)
def no_live_credentials_or_network(monkeypatch):
    monkeypatch.setattr("evo2_mvp.nvidia_encoder.load_api_key", lambda: FAKE_KEY)
    def forbidden_network(*args, **kwargs):
        raise AssertionError("Diagnostics unit tests must never make network requests")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden_network)


def failure(response):
    session = Session(response)
    encoder = NvidiaEvo2Encoder(api_schema="nim_v2", session=session, max_retries=0)
    with pytest.raises(NvidiaAPIError) as caught:
        encoder.encode("ACGT", 1)
    return caught.value, encoder, session


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key.lower()
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


def test_decoder_provider_error_is_visible_with_explicit_profile_repair_hint():
    error, encoder, session = failure(Response({"error": PROVIDER_ERROR}))
    diagnostics = error.diagnostics
    assert diagnostics["http_status"] == 422
    assert diagnostics["request_id"] == "test-request-422"
    assert "StripedHyena" in json.dumps(diagnostics["server_detail"])
    assert "decoder" in json.dumps(diagnostics["server_detail"])
    assert "nim_v1" in diagnostics["hint"]
    assert "blocks.20.mlp.l3" in diagnostics["hint"]
    assert "StripedHyena" in str(error)
    assert encoder.metadata["real_model_verified"] is False
    assert encoder.metadata["api_schema"] == "nim_v2"
    assert len(session.calls) == 1
    assert session.calls[0][1]["json"]["output_layers"] == ["decoder.layers.20.mlp.linear_fc2"]


def test_nested_json_diagnostics_whitelist_and_redact_all_credential_forms():
    other_bearer = "another-token-that-must-never-be-printed"
    other_nvapi = "nvapi-unit-test-third-private-value"
    response = Response({
        "error": PROVIDER_ERROR,
        "message": f"key echo {FAKE_KEY}; Bearer {other_bearer}; {other_nvapi}",
        "detail": [{"loc": ["body", "output_layers", 0],
                    "msg": f"Bearer {FAKE_KEY}", "type": "value_error",
                    "input": {"authorization": FAKE_KEY}, "ctx": {"message": FAKE_KEY}}],
        "headers": {"Authorization": "Bearer " + FAKE_KEY},
        "authorization": FAKE_KEY, "ctx": {"error": FAKE_KEY}, "input": FAKE_KEY,
    })
    error, _, _ = failure(response)
    rendered = str(error) + json.dumps(error.diagnostics, ensure_ascii=False)
    for secret in (FAKE_KEY, other_bearer, other_nvapi):
        assert secret not in rendered
    assert not {"headers", "input", "ctx", "authorization"} & set(nested_keys(error.diagnostics["server_detail"]))
    assert "decoder" in rendered
    assert "output_layers" in rendered


@pytest.mark.parametrize("payload,content_type", [
    (ValueError("invalid JSON " + FAKE_KEY), "application/json"),
    (None, "text/plain"),
])
def test_unstructured_or_invalid_json_body_is_ignored(payload, content_type):
    response = Response(payload, content_type=content_type)
    error, _, _ = failure(response)
    rendered = str(error) + json.dumps(error.diagnostics)
    assert FAKE_KEY not in rendered
    assert "unstructured echoed secret" not in rendered
    assert error.diagnostics["http_status"] == 422


def test_phase0_failure_report_retains_safe_api_error_without_fake_results(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "test_evo2.py"
    spec = importlib.util.spec_from_file_location("phase0_diagnostic_script_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = Session(Response({"error": PROVIDER_ERROR, "detail": {"message": FAKE_KEY}}))
    encoder = NvidiaEvo2Encoder(api_schema="nim_v2", session=session, max_retries=0)
    config = {"encoder": {"backend": "nvidia_api", "model": "arc/evo2-40b", "api_schema": "nim_v2"}}
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "create_encoder", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(module, "inspect_environment", lambda **kwargs: {"selected_backend_blockers": [], "selected_backend": "nvidia_api"})
    monkeypatch.setattr(sys, "argv", ["test_evo2.py", "--config", "unused-fixture.yaml", "--output", str(tmp_path)])
    assert module.main() == 2
    report = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["real_model_verified"] is False
    assert report["api_error"]["http_status"] == 422
    assert "decoder" in json.dumps(report["api_error"]["server_detail"])
    assert "nim_v1" in report["api_error"]["hint"]
    assert report["embedding_shape"] is None
    assert not list(tmp_path.glob("*.npy"))
    assert FAKE_KEY not in json.dumps(report) + capsys.readouterr().out
    assert len(session.calls) == 1


def test_hosted_environment_does_not_publish_legacy_local_1b_blockers(monkeypatch):
    from evo2_mvp import environment
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    report = environment.inspect_environment(backend="nvidia_api", model="arc/evo2-40b")
    assert report["official_model"] == "nvidia/arc/evo2-40b"
    assert report["selected_backend_blockers"] == []
    assert "official_1b_preflight_blockers" not in report
    assert report["ready_for_selected_backend_attempt"] is True
