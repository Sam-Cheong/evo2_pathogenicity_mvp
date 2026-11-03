"""Hosted adapter contract tests. Every response is a fixture, never inference evidence.

These tests never use a real API key or make network calls. Passing them verifies
client behavior, not NVIDIA availability, a particular remote model, or accuracy.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np
import pytest
import requests

from evo2_mvp.nvidia_encoder import (
    DEFAULT_ENDPOINT, MissingAPIKey, NvidiaAPIError, NvidiaEvo2Encoder, validate_endpoint,
)


FAKE_KEY = "unit-test-fake-key-no-provider-access"
LAYER = "blocks.20.mlp.l3"


def npz_bytes(**tensors):
    stream = io.BytesIO()
    np.savez(stream, **tensors)
    return stream.getvalue()


class Response:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = dict(headers or {})
        if payload is not None:
            self.headers.setdefault("Content-Type", "application/json")

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def tensor_response(array=None, key=None, status=200):
    if array is None:
        array = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    blob = npz_bytes(**{key or LAYER + ".output": array})
    return Response(status, {"data": base64.b64encode(blob).decode("ascii"), "elapsed_ms": 12.5})


class Session:
    def __init__(self, posts=(), gets=()):
        self.posts = list(posts)
        self.gets = list(gets)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        assert self.posts, "Unexpected POST; this unit-test session has no real network access"
        result = self.posts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        assert self.gets, "Unexpected GET; this unit-test session has no real network access"
        result = self.gets.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    # A local .env must never pull real credentials into this mocked test suite.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    monkeypatch.setattr("evo2_mvp.nvidia_encoder.time.sleep", lambda delay: None)


def test_request_contract_free_of_labels_and_secret_metadata():
    session = Session([tensor_response()])
    encoder = NvidiaEvo2Encoder(session=session)
    identity = encoder.metadata["feature_identity"]
    actual = encoder.encode("ACGT", 2)
    np.testing.assert_array_equal(actual, [6, 7, 8])
    assert encoder.last_token_shape == [1, 4, 3]
    assert encoder.last_request_info["server_elapsed_ms"] == 12.5
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", DEFAULT_ENDPOINT)
    assert kwargs["json"] == {"sequence": "ACGT", "output_layers": [LAYER]}
    assert kwargs["headers"]["Authorization"] == "Bearer " + FAKE_KEY
    assert kwargs["allow_redirects"] is False
    assert FAKE_KEY not in json.dumps(encoder.metadata)
    assert encoder.metadata["real_model_verified"] is True
    assert encoder.metadata["feature_identity"] == identity
    metadata = encoder.metadata
    metadata["layer"] = "tampered"
    assert encoder.metadata["layer"] == LAYER


def test_pair_features_keep_reference_alternate_order():
    ref = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    alt = ref + 2
    session = Session([tensor_response(ref), tensor_response(alt)])
    encoder = NvidiaEvo2Encoder(session=session)
    pair = encoder.encode_pair("ACGT", "ATGT", 1)
    np.testing.assert_array_equal(pair["ref"], [3, 4, 5])
    np.testing.assert_array_equal(pair["alt"], [5, 6, 7])
    np.testing.assert_array_equal(pair["delta"], [2, 2, 2])
    np.testing.assert_array_equal(pair["features"], [3, 4, 5, 5, 6, 7, 2, 2, 2])
    assert [call[2]["json"]["sequence"] for call in session.calls] == ["ACGT", "ATGT"]
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in pair.values())


@pytest.mark.parametrize("alternate,index", [("ACGT", 1), ("TCGA", 1), ("ATG", 1), ("ATGT", 2)])
def test_pair_rejects_non_snv_or_wrong_index_before_request(alternate, index):
    session = Session()
    with pytest.raises(ValueError, match="SNV"):
        NvidiaEvo2Encoder(session=session).encode_pair("ACGT", alternate, index)
    assert not session.calls


def test_ref_alt_hidden_dimensions_must_match():
    session = Session([tensor_response(np.zeros((1, 4, 3))), tensor_response(np.zeros((1, 4, 5)))])
    with pytest.raises(NvidiaAPIError, match="different hidden dimensions"):
        NvidiaEvo2Encoder(session=session).encode_pair("ACGT", "ATGT", 1)


@pytest.mark.parametrize("shape", [(1, 4, 3), (4, 1, 3)])
def test_supported_tensor_orientations(shape):
    array = np.arange(12, dtype=np.float32).reshape(shape)
    encoder = NvidiaEvo2Encoder(session=Session([tensor_response(array)]))
    np.testing.assert_array_equal(encoder.encode("ACGT", 1), [3, 4, 5])
    assert encoder.last_token_shape == [1, 4, 3]


def test_local_mean_pooling_clamps_sequence_edges():
    array = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    encoder = NvidiaEvo2Encoder(pooling="local_mean", local_radius=2,
                              session=Session([tensor_response(array)]))
    np.testing.assert_array_equal(encoder.encode("ACGT", 0), [3, 4, 5])


@pytest.mark.parametrize("array", [np.zeros((4, 3)), np.zeros((2, 4, 3)), np.zeros((1, 5, 3)),
                                  np.zeros((1, 4, 0)), np.full((1, 4, 3), np.nan),
                                  np.full((1, 4, 3), np.inf)])
def test_invalid_tensor_shape_or_nonfinite_rejected(array):
    encoder = NvidiaEvo2Encoder(session=Session([tensor_response(array)]))
    with pytest.raises(NvidiaAPIError):
        encoder.encode("ACGT", 1)
    assert encoder.metadata["real_model_verified"] is False


@pytest.mark.parametrize("sequence,index", [("", 0), ("acgt", 1), ("ANGT", 1), ("ACGT", -1),
                                           ("ACGT", 4), ("A" * 8193, 1)])
def test_invalid_sequence_never_reaches_endpoint(sequence, index):
    session = Session()
    with pytest.raises(ValueError):
        NvidiaEvo2Encoder(session=session).encode(sequence, index)
    assert session.calls == []


@pytest.mark.parametrize("endpoint", [
    "http://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward",
    "https://health.api.nvidia.com.evil.test/v1/biology/arc/evo2-40b/forward",
    "https://user:password@health.api.nvidia.com/v1/biology/arc/evo2-40b/forward",
    DEFAULT_ENDPOINT + "?token=anything", DEFAULT_ENDPOINT + "#fragment",
    "https://health.api.nvidia.com:444/v1/biology/arc/evo2-40b/forward",
    "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward",
])
def test_api_credentials_restricted_to_matching_official_https_endpoint(endpoint):
    with pytest.raises(ValueError, match="official HTTPS"):
        NvidiaEvo2Encoder(endpoint=endpoint, session=Session())


def test_hosted_one_b_model_is_not_silently_assumed():
    with pytest.raises(ValueError, match="1B"):
        validate_endpoint(DEFAULT_ENDPOINT, "arc/evo2-1b")


@pytest.mark.parametrize("key", [None, "", "   ", "your_nvidia_api_key", "replace_with_your_key", "nvapi-your-key-here"])
def test_missing_or_placeholder_key_is_clear_error(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("NVIDIA_API_KEY", key)
    session = Session()
    with pytest.raises(MissingAPIKey, match="NVIDIA_API_KEY is missing"):
        NvidiaEvo2Encoder(session=session)
    assert session.calls == []


@pytest.mark.parametrize("status,reason", [(401, "key rejected"), (403, "lacks endpoint access"),
                                          (422, "schema rejected")])
def test_http_errors_do_not_echo_provider_body_or_credentials(status, reason):
    session = Session([Response(status, {"error": "malicious echo " + FAKE_KEY})])
    encoder = NvidiaEvo2Encoder(session=session)
    with pytest.raises(NvidiaAPIError, match=reason) as caught:
        encoder.encode("ACGT", 1)
    assert FAKE_KEY not in str(caught.value)
    assert len(session.calls) == 1
    assert encoder.metadata["real_model_verified"] is False


def test_request_exception_redacts_secret_contained_in_exception():
    session = Session([requests.Timeout("Authorization Bearer " + FAKE_KEY)])
    with pytest.raises(NvidiaAPIError, match="failed or timed out") as caught:
        NvidiaEvo2Encoder(session=session).encode("ACGT", 1)
    assert FAKE_KEY not in str(caught.value)


def test_rate_limit_retries_are_bounded_and_retry_after_clamped(monkeypatch):
    delays = []
    monkeypatch.setattr("evo2_mvp.nvidia_encoder.time.sleep", delays.append)
    session = Session([Response(429, headers={"Retry-After": "99999"}),
                       Response(429, headers={"Retry-After": "not-a-number"}), tensor_response()])
    actual = NvidiaEvo2Encoder(session=session, max_retries=2).encode("ACGT", 1)
    np.testing.assert_array_equal(actual, [3, 4, 5])
    assert len(session.calls) == 3
    assert delays == [10., 1.]


def test_final_rate_limit_reports_quota_without_secret():
    session = Session([Response(429, {"error": FAKE_KEY}) for _ in range(3)])
    with pytest.raises(NvidiaAPIError, match="rate/quota limit") as caught:
        NvidiaEvo2Encoder(session=session, max_retries=2).encode("ACGT", 1)
    assert len(session.calls) == 3
    assert FAKE_KEY not in str(caught.value)


@pytest.mark.parametrize("payload", [{}, {"data": "not base64!"}, {"data": None}, ValueError("invalid JSON")])
def test_invalid_json_or_base64_payload(payload):
    encoder = NvidiaEvo2Encoder(session=Session([Response(payload=payload)]))
    with pytest.raises(NvidiaAPIError, match="documented JSON"):
        encoder.encode("ACGT", 1)


@pytest.mark.parametrize("blob", [b"not a zip", npz_bytes(wrong=np.zeros((1, 4, 3))),
    npz_bytes(**{LAYER + ".output": np.array([[[object()]]], dtype=object)})])
def test_malformed_or_pickle_npz_is_rejected(blob):
    response = Response(payload={"data": base64.b64encode(blob).decode("ascii")})
    with pytest.raises(NvidiaAPIError):
        NvidiaEvo2Encoder(session=Session([response])).encode("ACGT", 1)


def test_np_load_disables_pickle(monkeypatch):
    real_load, flags = np.load, []
    def audited_load(*args, **kwargs):
        flags.append(kwargs.get("allow_pickle"))
        return real_load(*args, **kwargs)
    monkeypatch.setattr("evo2_mvp.nvidia_encoder.np.load", audited_load)
    NvidiaEvo2Encoder(session=Session([tensor_response()])).encode("ACGT", 1)
    assert flags and all(flag is False for flag in flags)


def test_missing_and_ambiguous_layer_keys_require_explicit_tensor_key():
    array = np.zeros((1, 4, 3), np.float32)
    blob = npz_bytes(**{LAYER: array, LAYER + ".output": array + 1})
    response = Response(content=blob, headers={"Content-Type": "application/octet-stream"})
    with pytest.raises(NvidiaAPIError, match="missing/ambiguous"):
        NvidiaEvo2Encoder(session=Session([response])).encode("ACGT", 1)
    encoder = NvidiaEvo2Encoder(tensor_key=LAYER, session=Session([response]))
    np.testing.assert_array_equal(encoder.encode("ACGT", 1), [0, 0, 0])


def test_explicit_custom_tensor_key_and_profile_selection():
    layer = "decoder.layers.20.mlp.linear_fc2"
    encoder = NvidiaEvo2Encoder(api_schema="nim_v2", tensor_key="explicit_hidden",
        session=Session([tensor_response(key="explicit_hidden")]))
    encoder.encode("ACGT", 1)
    assert encoder.metadata["layer"] == layer
    assert encoder.session.calls[0][2]["json"]["output_layers"] == [layer]
    with pytest.raises(ValueError, match="schema profile"):
        NvidiaEvo2Encoder(api_schema="nim_v2", layer=LAYER, session=Session())


def test_direct_binary_npz_response():
    blob = npz_bytes(**{LAYER + ".output": np.arange(12, dtype=np.float32).reshape(1, 4, 3)})
    response = Response(content=blob, headers={"Content-Type": "application/octet-stream"})
    np.testing.assert_array_equal(NvidiaEvo2Encoder(session=Session([response])).encode("ACGT", 1), [3, 4, 5])


def outer_zip(inner, names=("result.npz",)):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in names:
            archive.writestr(name, inner)
    return stream.getvalue()


def test_outer_zip_with_single_npz_is_decoded_in_memory(monkeypatch):
    def forbid_extract(*args, **kwargs):
        raise AssertionError("Archive tensors must never be extracted to filesystem")
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbid_extract)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbid_extract)
    blob = outer_zip(npz_bytes(**{LAYER: np.arange(12, dtype=np.float32).reshape(1, 4, 3)}))
    response = Response(content=blob, headers={"Content-Type": "application/zip"})
    np.testing.assert_array_equal(NvidiaEvo2Encoder(session=Session([response])).encode("ACGT", 1), [3, 4, 5])


@pytest.mark.parametrize("names", [("one.npz", "two.npz"), ("not_numpy.txt",)])
def test_unsafe_or_ambiguous_outer_zip_rejected(names):
    blob = outer_zip(npz_bytes(**{LAYER: np.zeros((1, 4, 3))}), names)
    response = Response(content=blob, headers={"Content-Type": "application/zip"})
    with pytest.raises(NvidiaAPIError):
        NvidiaEvo2Encoder(session=Session([response])).encode("ACGT", 1)


def test_async_202_polls_only_official_status_endpoint():
    request_id = "1234-5678-abcd"
    accepted = Response(202, headers={"NVCF-REQID": request_id})
    session = Session([accepted], [accepted, tensor_response()])
    np.testing.assert_array_equal(NvidiaEvo2Encoder(session=session).encode("ACGT", 1), [3, 4, 5])
    gets = [call for call in session.calls if call[0] == "GET"]
    assert len(gets) == 2
    for _, url, kwargs in gets:
        assert url == "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/" + request_id
        assert kwargs["headers"]["Authorization"] == "Bearer " + FAKE_KEY
        assert kwargs["allow_redirects"] is False


@pytest.mark.parametrize("request_id", [None, "", "../redirect", "abc?x=bad", "https://evil.test"])
def test_async_request_id_cannot_control_request_url(request_id):
    headers = {} if request_id is None else {"NVCF-REQID": request_id}
    session = Session([Response(202, headers=headers)])
    with pytest.raises(NvidiaAPIError, match="valid NVCF-REQID"):
        NvidiaEvo2Encoder(session=session).encode("ACGT", 1)
    assert len(session.calls) == 1


def test_async_polling_timeout_is_bounded(monkeypatch):
    clocks = iter([0., 2.])
    monkeypatch.setattr("evo2_mvp.nvidia_encoder.time.monotonic", lambda: next(clocks))
    session = Session([Response(202, headers={"NVCF-REQID": "1234"})])
    with pytest.raises(NvidiaAPIError, match="poll_timeout_seconds"):
        NvidiaEvo2Encoder(session=session, poll_timeout_seconds=1).encode("ACGT", 1)


def test_302_result_download_has_no_authorization():
    location = "https://example-cdn.test/result/fixture-signed-url"
    blob = npz_bytes(**{LAYER: np.zeros((1, 4, 3), dtype=np.float32)})
    session = Session([Response(302, headers={"Location": location})],
                      [Response(content=blob, headers={"Content-Type": "application/octet-stream"})])
    np.testing.assert_array_equal(NvidiaEvo2Encoder(session=session).encode("ACGT", 1), [0, 0, 0])
    _, url, kwargs = session.calls[1]
    assert url == location
    assert "Authorization" not in kwargs.get("headers", {})
    assert kwargs["allow_redirects"] is False


def test_causal_sequence_score_uses_next_character_ascii_tokens():
    ref = np.zeros((1, 4, 512), dtype=np.float32)
    alt = ref.copy()
    for logits, first_token, first_value in [(ref, "C", 4), (alt, "T", 1)]:
        logits[0, 0, ord(first_token)] = first_value
        logits[0, 1, ord("G")] = 3
        logits[0, 2, ord("T")] = 2
        logits[0, 3, ord("A")] = 99  # Last token has no next-base target: excluded.
    session = Session([tensor_response(ref, "unembed.output"), tensor_response(alt, "unembed.output")])
    scores = NvidiaEvo2Encoder(session=session).score_pair("ACGT", "ATGT")
    expected_ref = float(np.mean([x-np.log(np.exp(x)+511) for x in (4, 3, 2)]))
    expected_alt = float(np.mean([x-np.log(np.exp(x)+511) for x in (1, 3, 2)]))
    assert scores["mean_log_likelihood_ref"] == pytest.approx(expected_ref)
    assert scores["mean_log_likelihood_alt"] == pytest.approx(expected_alt)
    assert scores["delta_mean_log_likelihood"] == pytest.approx(expected_alt-expected_ref)
    assert scores["interpretation"] == "sequence_likelihood_only_not_pathogenicity_probability"
    assert [call[2]["json"]["output_layers"] for call in session.calls] == [["unembed"], ["unembed"]]


def test_scoring_rejects_unverified_vocabulary_shape():
    encoder = NvidiaEvo2Encoder(session=Session([tensor_response(np.zeros((1, 4, 5)), "unembed")]))
    with pytest.raises(NvidiaAPIError, match="512"):
        encoder.score_pair("ACGT", "ATGT")
