"""Weight-free HyenaDNA contracts; these tests do not measure pathogenicity.

Only the explicitly opted-in integration test reads real pretrained weights.
No test calls Evo 2, NVIDIA, or the Hugging Face network API.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from evo2_mvp.hyenadna import HyenaDNAStudent
from evo2_mvp.models import TinyStudent, build_student, load_checkpoint, one_hot


# Fixed architecture values from the pinned official d256 checkpoint config.
# Construction initializes test-only weights and never reads a work/ checkpoint.
ENCODER_CONFIG = {
    "vocab_size": 12, "d_model": 256, "d_inner": 1024,
    "use_bias": True, "train_freq": True, "max_seq_len": 1026,
    "emb_dim": 5, "n_layer": 2, "num_inner_mlps": 2,
    "hyena_order": 2, "short_filter_order": 3, "filter_order": 64,
    "activation_freq": 10, "embed_dropout": 0.1,
    "hyena_dropout": 0.0, "hyena_filter_dropout": 0.0,
    "layer_norm_epsilon": 1e-5, "initializer_range": 0.02,
    "pad_vocab_size_multiple": 8, "pad_token_id": 4,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def modest_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def student():
    torch.manual_seed(173)
    return HyenaDNAStudent(copy.deepcopy(ENCODER_CONFIG), dropout=0.0).eval()


def snv_contexts():
    reference = "ACGT" * 64
    alternate = reference[:128] + "G" + reference[129:]
    return one_hot([reference]), one_hot([alternate])


def test_token_ids_preserve_every_base_without_special_tokens(student):
    values = one_hot(["ACGTAC", "TGCATG"])
    ids = student.token_ids(values)
    assert ids.dtype == torch.long
    assert ids.shape == (2, 6)
    assert torch.equal(ids, torch.tensor([[7, 8, 9, 10, 7, 8], [10, 9, 8, 7, 10, 9]]))
    ref, alt = snv_contexts()
    ref_ids, alt_ids = student.token_ids(ref), student.token_ids(alt)
    assert ref_ids.shape == (1, 256)
    assert torch.nonzero(ref_ids != alt_ids).tolist() == [[0, 128]]
    assert (ref_ids[0, 128].item(), alt_ids[0, 128].item()) == (7, 9)


def test_center_is_the_actual_hidden_token_not_a_sequence_pool(student):
    ref, _ = snv_contexts()
    with torch.no_grad():
        hidden = student.encoder(student.token_ids(ref), return_dict=True).last_hidden_state
        center = student.encode(ref)
        explicit = student.encode(ref, variant_index=17)
    assert hidden.shape == (1, 256, 256)
    torch.testing.assert_close(center, hidden[:, 128, :], rtol=0, atol=0)
    torch.testing.assert_close(explicit, hidden[:, 17, :], rtol=0, atol=0)
    assert not torch.allclose(center, hidden[:, 127, :])
    assert not torch.allclose(center, hidden.mean(dim=1))


def test_causal_center_detects_snv_and_ignores_future_bases(student):
    ref, alt = snv_contexts()
    reference = "ACGT" * 64
    # Change every base strictly to the right of the variant, retaining position 128.
    complement = str.maketrans("ACGT", "TGCA")
    future_changed = one_hot([reference[:129] + reference[129:].translate(complement)])
    with torch.no_grad():
        ref_embedding = student.encode(ref)
        alt_embedding = student.encode(alt)
        future_embedding = student.encode(future_changed)
        ref_prefix = student.encode(ref, variant_index=127)
        alt_prefix = student.encode(alt, variant_index=127)
        logits, fused = student(ref, alt)
    assert logits.shape == (1,) and fused.shape == (1, 768)
    assert torch.isfinite(logits).all() and torch.isfinite(fused).all()
    assert not torch.allclose(ref_embedding, alt_embedding, rtol=1e-5, atol=1e-5)
    # FFT rounding can produce tiny errors; causal independence need not be bitwise.
    torch.testing.assert_close(ref_embedding, future_embedding, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_prefix, alt_prefix, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[:, :256], ref_embedding, rtol=0, atol=0)
    torch.testing.assert_close(fused[:, 256:512], alt_embedding, rtol=0, atol=0)
    torch.testing.assert_close(fused[:, 512:], alt_embedding - ref_embedding, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["zero_column", "fractional", "nan", "too_long", "wrong_channels"])
def test_invalid_dna_cannot_silently_become_padding_or_get_truncated(student, case):
    values = one_hot(["ACGT" * 4])
    if case == "zero_column":
        values[:, :, 4] = 0
    elif case == "fractional":
        values[:, :, 4] = 0
        values[:, :2, 4] = 0.5
    elif case == "nan":
        values[0, 0, 0] = float("nan")
    elif case == "too_long":
        values = one_hot(["A" * 1025])
    else:
        values = values[:, :3]
    with pytest.raises(ValueError):
        student.token_ids(values)


@pytest.mark.parametrize("index", [-1, 256, True, 1.5])
def test_variant_index_must_address_an_actual_base(student, index):
    ref, _ = snv_contexts()
    with pytest.raises(ValueError, match="variant_index"):
        student.encode(ref, variant_index=index)


def test_classifier_loss_reaches_the_encoder_and_stays_below_2m(student):
    ref, alt = snv_contexts()
    # eval disables dropout but does not freeze the trainable backbone.
    student.zero_grad(set_to_none=True)
    logits, _ = student(ref, alt)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    embedding = student.encoder.backbone.embeddings.word_embeddings.weight
    assert embedding.requires_grad and embedding.grad is not None
    assert torch.isfinite(embedding.grad).all() and embedding.grad.abs().sum() > 0
    layer_grads = [p.grad for name, p in student.encoder.named_parameters()
                   if name.startswith("backbone.layers.0.") and p.grad is not None]
    assert layer_grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in layer_grads)
    assert sum(p.numel() for p in student.parameters()) == 1_854_465
    assert sum(p.numel() for p in student.parameters()) < 2_000_000
    assert student.feature_dim == 768


def test_hyenadna_factory_roundtrip_is_offline_and_teacher_independent(student, tmp_path):
    kwargs = {"encoder_config": copy.deepcopy(ENCODER_CONFIG), "head_dims": [256, 64],
              "dropout": 0.0, "pooling": "variant_center"}
    path = tmp_path / "test_only_hyenadna.pt"
    torch.save({"kind": "student", "architecture": "hyenadna_d256",
                "model_kwargs": kwargs, "model_state": student.state_dict()}, path)
    ref, alt = snv_contexts()
    with torch.no_grad():
        expected_logits, expected_features = student(ref, alt)
        actual_logits, actual_features = build_student(load_checkpoint(path))(ref, alt)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    torch.save({"logits": expected_logits, "features": expected_features}, tmp_path / "expected.pt")

    code = """
import socket, sys
def deny_network(*args, **kwargs):
    raise AssertionError('Student checkpoint reload attempted network access')
socket.socket.connect = deny_network
import torch
torch.set_num_threads(1)
from evo2_mvp.models import build_student, load_checkpoint, one_hot
model = build_student(load_checkpoint(sys.argv[1]))
reference = 'ACGT' * 64
alternate = reference[:128] + 'G' + reference[129:]
with torch.no_grad():
    logits, features = model(one_hot([reference]), one_hot([alternate]))
expected = torch.load(sys.argv[2], weights_only=True)
torch.testing.assert_close(logits, expected['logits'], rtol=0, atol=0)
torch.testing.assert_close(features, expected['features'], rtol=0, atol=0)
assert 'evo2' not in sys.modules
assert 'evo2_mvp.nvidia_encoder' not in sys.modules
print('offline_hyenadna_student_ok')
"""
    environment = dict(os.environ)
    for name in ("NVIDIA_API_KEY", "NVCF_RUN_KEY", "NGC_API_KEY"):
        environment.pop(name, None)
    root = Path(__file__).resolve().parents[1]
    environment.update({"PYTHONPATH": str(root / "src"), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(tmp_path / "empty_hf_cache")})
    result = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "expected.pt")],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "offline_hyenadna_student_ok" in result.stdout


@pytest.mark.parametrize("explicit_architecture", [False, True])
def test_existing_cnn_checkpoints_remain_compatible(explicit_architecture, tmp_path):
    torch.manual_seed(19)
    model = TinyStudent(embedding_dim=8, channels=[4, 8], dropout=0.0).eval()
    checkpoint = {"kind": "student", "model_kwargs": {"embedding_dim": 8, "channels": [4, 8],
                                                        "dropout": 0.0},
                  "model_state": model.state_dict()}
    if explicit_architecture:
        checkpoint["architecture"] = "cnn"
    path = tmp_path / "legacy_cnn.pt"
    torch.save(checkpoint, path)
    ref, alt = snv_contexts()
    with torch.no_grad():
        expected = model(ref, alt)
        actual = build_student(load_checkpoint(path))(ref, alt)
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def test_unknown_architecture_never_falls_back_to_a_random_cnn():
    with pytest.raises(ValueError, match="Unsupported student architecture"):
        build_student({"kind": "student", "architecture": "misspelled_hyena", "model_kwargs": {}})


@pytest.mark.hyenadna_integration
@pytest.mark.skipif(os.getenv("RUN_HYENADNA_INTEGRATION") != "1",
                    reason="Real pinned HyenaDNA weights not requested; set RUN_HYENADNA_INTEGRATION=1")
def test_real_pinned_pretrained_backbone_loads_strictly_and_preserves_center():
    from evo2_mvp.hyenadna import (PRETRAINED_ID, PRETRAINED_REVISION,
                                  initialize_pretrained_student, state_digest)

    root = Path(__file__).resolve().parents[1]
    directory = Path(os.getenv("HYENADNA_PRETRAINED_DIR", str(root / "work" / "hyenadna_d256_pretrained")))
    model, kwargs, metadata = initialize_pretrained_student({"pretrained_dir": str(directory),
                                                            "dropout": 0.0}, window=256)
    model.eval()
    assert metadata["repo_id"] == PRETRAINED_ID and metadata["revision"] == PRETRAINED_REVISION
    assert metadata["strict_load"] is True and metadata["pretrained_encoder_loaded"] is True
    assert metadata["restored_shared_aliases"]
    assert metadata["discarded_pretrained_tensors"] == ["lm_head.weight"]
    assert metadata["encoder_initial_state_sha256"] == state_digest(model.encoder.state_dict())
    assert metadata["parameter_count"] == sum(p.numel() for p in model.parameters()) == 1_854_465
    assert kwargs["pooling"] == "variant_center"
    ref, alt = snv_contexts()
    with torch.no_grad():
        ref_embedding, alt_embedding = model.encode(ref), model.encode(alt)
        _, features = model(ref, alt)
    assert ref_embedding.shape == alt_embedding.shape == (1, 256)
    assert features.shape == (1, 768) and torch.isfinite(features).all()
    assert not torch.allclose(ref_embedding, alt_embedding)
    # This validates the pretrained backbone, not a clinically trained head.
