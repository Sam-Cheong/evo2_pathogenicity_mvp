"""CPU value/gradient parity with fixed, SHA256-verified original Savanna bodies.

Missing optional public source files skip the suite. A wrong source hash fails.
Run scripts/prepare_sh2_reference.py --download --self-check to fetch/require them.
No Evo2 service, pretrained weights, labels, or training data are used here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from evo2_mvp.stripedhyena2 import (
    ExplicitDecayFilter, GatedMLP, ImplicitModalFilter, ReferenceRMSNorm,
    RotaryCausalAttention, causal_convolution, gated_hyena,
)

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-5, 1e-4
GRAD_ATOL, GRAD_RTOL = 5e-5, 3e-4


@pytest.fixture(scope="module")
def official():
    script = ROOT / "scripts" / "prepare_sh2_reference.py"
    if not script.exists():
        pytest.skip("Optional pinned official source loader is absent")
    spec = importlib.util.spec_from_file_location("sh2_official_reference", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_dir = ROOT / "work" / "sh2_reference"
    missing = [p for p in module.FILES if not (reference_dir / p).is_file()]
    if missing:
        pytest.skip("Download pinned official reference sources first: " + ", ".join(missing))
    # Integrity failures are genuine failures and must never become skips.
    for relative in module.FILES:
        module._verify(relative, (reference_dir / relative).read_bytes())
    return lambda name: module.load_official_function(name, reference_dir)


def _tensor(*shape):
    return torch.randn(*shape, dtype=torch.float32, requires_grad=True)


def _compare(actual, expected, variables):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    probe = torch.linspace(-0.3, 0.7, actual.numel()).reshape(actual.shape)
    grads_actual = torch.autograd.grad((actual * probe).sum(), variables)
    grads_expected = torch.autograd.grad((expected * probe).sum(), variables)
    for actual_grad, expected_grad in zip(grads_actual, grads_expected):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)


@pytest.mark.parametrize("filter_length", [3, 9, 23])
def test_causal_convolution_matches_original_fft_value_and_gradients(official, filter_length):
    torch.manual_seed(81)
    values, impulse = _tensor(2, 4, 23), _tensor(4, filter_length)
    expected = official("fftconv_func")(values, impulse, torch.zeros(4), None, gelu=False)
    _compare(causal_convolution(values, impulse), expected, (values, impulse))


def test_se_original_gates_short_convolution_and_gradients(official):
    torch.manual_seed(82)
    x1, x2, values, impulse = _tensor(2, 4, 23), _tensor(2, 4, 23), _tensor(2, 4, 23), _tensor(4, 7)
    original_conv = official("ParallelCausalDepthwiseConv1d.forward")
    # Original conv stores cross-correlation weights; our h stores lag-zero first.
    state_conv = SimpleNamespace(short_conv_weight=impulse.flip(-1)[:, None], kernel_size=7,
        group_dim=1, use_cp_hyena=False, use_fast_causal_conv=False, conv_groups=4)
    state_se = SimpleNamespace(use_custom_hyena_mlp_kernel=False,
        use_custom_hyena_short_kernel=False, use_cgcg_mlp=False, use_cgcg_short=False,
        pregate=True, postgate=True, short_conv=lambda u: original_conv(state_conv, u))
    shape = lambda x: x.transpose(1, 2)[..., None]
    expected = official("ParallelShortHyenaOperator.forward")(state_se, shape(x1), shape(x2), shape(values)).transpose(1, 2)
    _compare(gated_hyena(x1, x2, values, impulse), expected, (x1, x2, values, impulse))


def test_mr_li_gate_and_diagonal_match_original_fft_gradients(official):
    torch.manual_seed(83)
    x1, x2, values, impulse, diagonal = _tensor(2, 4, 23), _tensor(2, 4, 23), _tensor(2, 4, 23), _tensor(4, 9), _tensor(4)
    expected = x1 * official("fftconv_func")(x2 * values, impulse, diagonal, None, gelu=False)
    _compare(gated_hyena(x1, x2, values, impulse, diagonal), expected, (x1, x2, values, impulse, diagonal))


@pytest.mark.parametrize("length", [5, 17, 23])
def test_mr_original_explicit_filter_values_and_gradients(official, length):
    torch.manual_seed(84)
    layer = ExplicitDecayFilter(4, length=17)
    expected = official("ExplicitSingleDecayFilter.filter")(layer, min(length, 17))
    _compare(layer(length), expected, (layer.h,))
    # Strong decay is an explicit custom setting supported by the official class.
    expected_decay = torch.exp(-torch.logspace(0, 2, 4)[:, None] * torch.linspace(0, 1, 17)[None])
    torch.testing.assert_close(layer.decay, expected_decay)


def test_li_original_compute_filter_values_and_gradients(official):
    torch.manual_seed(85)
    layer = ImplicitModalFilter(4, order=3)
    time = torch.arange(23, dtype=torch.float32)[None, None]
    expected, extra = official("ImplicitModalFilter.compute_filter")(layer, 23, time)
    assert extra is None and expected.shape == (1, 4, 23)
    _compare(layer(23), expected[0], (layer.p, layer.gamma, layer.R))


def test_rmsnorm_original_epsilon_placement_and_gradients(official):
    torch.manual_seed(86)
    layer = ReferenceRMSNorm(4, eps=1e-6)
    values = _tensor(2, 23, 4)
    state = SimpleNamespace(p=-1, d=4, eps=layer.eps, scale=layer.scale, bias=False)
    expected = official("RMSNorm.forward")(state, values)
    _compare(layer(values), expected, (values, layer.scale))
    # Near-zero values distinguish eps outside sqrt from standard torch RMSNorm.
    near_zero = torch.full((1, 1, 4), 1e-8)
    torch.testing.assert_close(layer(near_zero), official("RMSNorm.forward")(state, near_zero), atol=1e-8, rtol=1e-6)


def test_swiglu_original_parallel_glu_body_and_gradients(official):
    torch.manual_seed(87)
    layer = GatedMLP(4, 8)
    values = _tensor(2, 23, 4)
    state = SimpleNamespace(should_permute=False, seq_dim=1, activation_func=F.silu,
        global_config=SimpleNamespace(debug_print=False, seq_length=23, sequence_parallel=False),
        w1=lambda x: (layer.l1(x), None), w2=lambda x: (layer.l2(x), None), w3=lambda x: (layer.l3(x), None))
    expected, bias = official("ParallelGLU.forward_func")(state, values)
    assert bias is None
    _compare(layer(values), expected, (values, *layer.parameters()))


def test_rotary_matches_original_rope_body_and_gradients(official):
    torch.manual_seed(88)
    layer = RotaryCausalAttention(24, 3, dropout=0)
    query, key = _tensor(2, 3, 23, 8), _tensor(2, 3, 23, 8)
    phase = torch.outer(torch.arange(23).float(), layer.inverse_frequency)
    phase = torch.cat((phase, phase), dim=-1)[:, None, None]
    query_ref, key_ref = official("apply_rotary_pos_emb_torch")(query.permute(2, 0, 1, 3), key.permute(2, 0, 1, 3), phase.cos(), phase.sin())
    actual = torch.stack((layer.rotary(query), layer.rotary(key)))
    expected = torch.stack((query_ref.permute(1, 2, 0, 3), key_ref.permute(1, 2, 0, 3)))
    _compare(actual, expected, (query, key))
