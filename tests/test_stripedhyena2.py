"""Structural contracts for the custom small SH2 reference architecture."""
import sys

import pytest
import torch

from evo2_mvp.models import build_student, one_hot
from evo2_mvp.stripedhyena2 import StripedHyena2Student


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return StripedHyena2Student().eval()


def test_actual_multiscale_architecture_and_inference_size(model):
    assert [b.kind for b in model.encoder.blocks] == ["se", "mr", "li", "attention"]
    assert sum(p.numel() for p in model.parameters()) == 1_505_281
    assert sum(p.numel() for p in StripedHyena2Student(head_kind="mlp").parameters()) < 2_000_000
    assert all(b.mixer.input_convolution.groups == 576 for b in model.encoder.blocks[:3])


def test_center_snv_alignment_and_causality(model):
    sequence = "ACGT" * 64
    alt = sequence[:128] + "C" + sequence[129:]
    future = sequence[:129] + "A" * 127
    with torch.no_grad():
        original = model.encode(one_hot([sequence]))
        mutant = model.encode(one_hot([alt]))
        future_only = model.encode(one_hot([future]))
        _, features = model(one_hot([sequence]), one_hot([alt]))
    assert original.shape == (1, 192) and features.shape == (1, 576)
    assert torch.isfinite(features).all() and not torch.allclose(original, mutant)
    # FFT roundoff can depend on suffix values although the convolution is causal.
    torch.testing.assert_close(original, future_only, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(features[:, 384:], mutant - original)


def test_backward_reaches_each_operator(model):
    ref = one_hot(["ACGT" * 64])
    alt = one_hot(["C" + "ACGT"[1:] + "ACGT" * 63])
    logits, features = model(ref, alt)
    (logits.square().mean() + (features * torch.linspace(-1, 1, 576)).square().mean()).backward()
    for block in model.encoder.blocks:
        gradients = [p.grad for p in block.mixer.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.parametrize("bad", ["zero", "fractional", "nonfinite", "wrong_length"])
def test_strict_dna_input(model, bad):
    values = one_hot(["ACGT" * 64])
    if bad == "zero":
        values[:, :, 0] = 0
    elif bad == "fractional":
        values[:, :, 0] = .25
    elif bad == "nonfinite":
        values[0, 0, 0] = float("nan")
    else:
        values = values[:, :, :-1]
    with pytest.raises(ValueError):
        model.encode(values)


def test_factory_reload_contains_encoder_and_train_only_head_scaler(model, monkeypatch):
    model.feature_mean.fill_(.17)
    model.feature_scale.fill_(1.3)
    checkpoint = {"kind": "student", "architecture": "stripedhyena2_tiny", "model_state": model.state_dict(),
                  "model_kwargs": {"head_kind": "linear"}}
    monkeypatch.setitem(sys.modules, "evo2", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    restored = build_student(checkpoint)
    values = one_hot(["ACGT" * 64])
    with torch.no_grad():
        torch.testing.assert_close(model(values, values)[0], restored(values, values)[0], atol=0, rtol=0)
