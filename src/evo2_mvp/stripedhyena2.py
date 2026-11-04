"""Small, trainable StripedHyena 2 reference variant for the cache-only pilot.

Custom scale/schedule: SE -> MR -> LI -> causal rotary attention. The operators
follow Savanna's pinned equations (see docs/SH2_ARCHITECTURE_AUDIT.md); this is
neither the official Evo 2 checkpoint nor pretrained HyenaDNA. It starts from
seeded random weights and learns from genuine cached Evo 2 representations.
FP32 PyTorch reference operations deliberately replace GPU/distributed kernels.

Mathematical reference: Zymrael/savanna (Apache-2.0), Michael Poli, Eric Nguyen,
commit 7b819b8609ce0b29deff79358a294b59d8299d07. Original reference sources and
license are downloaded by scripts/prepare_sh2_reference.py for parity tests.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "stripedhyena2_tiny"
REFERENCE_REVISION = "7b819b8609ce0b29deff79358a294b59d8299d07"


def causal_convolution(values, impulse):
    """Depthwise causal linear convolution; impulse[:, 0] is lag zero."""
    if values.ndim != 3 or impulse.ndim != 2 or values.shape[1] != impulse.shape[0]:
        raise ValueError("Expected values (B,D,L) and impulse (D,K)")
    length = values.shape[-1]
    size = 2 * length
    kernel = impulse[..., :length]
    # Same zero-padding and normalization as the official fftconv_func.
    return torch.fft.irfft(torch.fft.rfft(values, n=size) * torch.fft.rfft(kernel, n=size),
                           n=size)[..., :length]


def gated_hyena(x1, x2, values, impulse, diagonal=None):
    product = x2 * values
    mixed = causal_convolution(product, impulse)
    if diagonal is not None:
        mixed = mixed + product * diagonal[:, None]
    return x1 * mixed


class ReferenceRMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.width, self.eps = width, eps

    def forward(self, values):
        return self.scale * values / (values.norm(2, dim=-1, keepdim=True) * self.width ** -0.5 + self.eps)


class ExplicitDecayFilter(nn.Module):
    """Hyena-MR: learned finite impulse coefficients times fixed decay."""
    def __init__(self, width, length=64):
        super().__init__()
        self.h = nn.Parameter(torch.randn(width, length) / math.sqrt(length))
        decay = torch.exp(-torch.logspace(0, 2, width)[:, None] * torch.linspace(0, 1, length)[None])
        self.register_buffer("decay", decay)

    def forward(self, length):
        return (self.h * self.decay)[..., :min(length, self.h.shape[-1])]


class ImplicitModalFilter(nn.Module):
    """Hyena-LI: h_d(t) = sum_o R_do exp(-exp(p_do+gamma_do)*t)."""
    def __init__(self, width, order=16):
        super().__init__()
        self.p = nn.Parameter(-torch.ones(width, order))
        self.gamma = nn.Parameter((torch.rand(width, order) * .09 + .01).log())
        self.R = nn.Parameter(.1 * torch.randn(width, order) / math.sqrt(order))

    def forward(self, length):
        time = torch.arange(length, dtype=torch.float32, device=self.p.device)[None, None]
        logp = -torch.exp(self.p.float())
        rates = logp * torch.exp(self.gamma.float())
        modes = torch.exp(rates[..., None] * time)
        return torch.einsum("do,dot->dt", self.R.float(), modes)


class MultiScaleHyena(nn.Module):
    def __init__(self, width, kind, short_length=3, medium_length=64, modal_order=16):
        super().__init__()
        if kind not in ("se", "mr", "li"):
            raise ValueError("Unknown Hyena operator")
        self.kind, self.width = kind, width
        self.in_projection = nn.Linear(width, 3 * width, bias=False)
        self.input_convolution = nn.Conv1d(3 * width, 3 * width, short_length, groups=3 * width, bias=False)
        self.out_projection = nn.Linear(width, width, bias=False)
        if kind == "se":
            self.short_impulse = nn.Parameter(torch.randn(width, short_length) / math.sqrt(short_length))
            self.register_parameter("diagonal", None)
        else:
            self.filter = ExplicitDecayFilter(width, medium_length) if kind == "mr" else ImplicitModalFilter(width, modal_order)
            self.diagonal = nn.Parameter(torch.ones(width))

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        projected = self.in_projection(hidden).transpose(1, 2)
        projected = self.input_convolution(F.pad(projected, (self.input_convolution.kernel_size[0] - 1, 0)))
        # Official interleaved three-way split, num_groups=width and group_dim=1.
        projected = projected.transpose(1, 2).reshape(batch, length, self.width, 3)
        x1, x2, values = (projected[..., index].transpose(1, 2) for index in range(3))
        impulse = self.short_impulse if self.kind == "se" else self.filter(length)
        mixed = gated_hyena(x1, x2, values, impulse, self.diagonal)
        return self.out_projection(mixed.transpose(1, 2))


class RotaryCausalAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("Attention requires an even integral head width")
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.register_buffer("inverse_frequency", 1 / (1_000_000 ** (torch.arange(0, self.head_width, 2).float() / self.head_width)))

    def rotary(self, value):
        phase = torch.outer(torch.arange(value.shape[-2], device=value.device).float(), self.inverse_frequency)
        phase = torch.cat((phase, phase), dim=-1).to(value.dtype)
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return value * phase.cos() + rotated * phase.sin()

    def forward(self, hidden):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        mixed = F.scaled_dot_product_attention(self.rotary(query), self.rotary(key), value,
                is_causal=True, dropout_p=self.dropout if self.training else 0.)
        return self.out(mixed.transpose(1, 2).reshape(batch, length, width))


class GatedMLP(nn.Module):
    def __init__(self, width, inner):
        super().__init__()
        self.l1 = nn.Linear(width, inner, bias=False)
        self.l2 = nn.Linear(width, inner, bias=False)
        self.l3 = nn.Linear(inner, width, bias=False)

    def forward(self, hidden):
        return self.l3(F.silu(self.l1(hidden)) * self.l2(hidden))


class ReferenceBlock(nn.Module):
    def __init__(self, width, inner, heads, kind, dropout):
        super().__init__()
        self.kind = kind
        self.pre_norm, self.post_norm = ReferenceRMSNorm(width), ReferenceRMSNorm(width)
        self.mixer = RotaryCausalAttention(width, heads, dropout) if kind == "attention" else MultiScaleHyena(width, kind)
        self.mlp, self.dropout = GatedMLP(width, inner), nn.Dropout(dropout)

    def forward(self, hidden):
        hidden = hidden + self.dropout(self.mixer(self.pre_norm(hidden)))
        return hidden + self.dropout(self.mlp(self.post_norm(hidden)))


class StripedHyena2Encoder(nn.Module):
    def __init__(self, d_model=192, ffn_dim=384, num_heads=6, window=256, dropout=.1):
        super().__init__()
        if not isinstance(window, int) or not 3 <= window <= 1024:
            raise ValueError("window must be in 3..1024")
        self.window, self.d_model = window, d_model
        # Four DNA bases, not Evo 2's ASCII tokenizer; no special/padding tokens.
        self.embedding = nn.Embedding(4, d_model)
        self.blocks = nn.ModuleList([ReferenceBlock(d_model, ffn_dim, num_heads, kind, dropout)
                                     for kind in ("se", "mr", "li", "attention")])
        self.final_norm = ReferenceRMSNorm(d_model)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1] != 4 or sequence.shape[-1] != self.window:
            raise ValueError("Expected strict (B,4,configured_window) DNA input")
        if not torch.isfinite(sequence).all() or not ((sequence == 0) | (sequence == 1)).all() or not (sequence.sum(1) == 1).all():
            raise ValueError("Expected finite strict A/C/G/T one-hot; no padding or truncation")
        hidden = self.embedding(sequence.argmax(1).long())
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class StripedHyena2Student(nn.Module):
    def __init__(self, d_model=192, ffn_dim=384, num_heads=6, window=256, head_kind="linear", dropout=.1):
        super().__init__()
        self.encoder = StripedHyena2Encoder(d_model, ffn_dim, num_heads, window, dropout)
        self.feature_dim, self.max_length, self.head_kind = 3 * d_model, window, head_kind
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        if head_kind == "linear":
            self.head = nn.Linear(self.feature_dim, 1)
        elif head_kind == "mlp":
            self.head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        else:
            raise ValueError("head_kind must be linear or mlp")
        if sum(p.numel() for p in self.parameters()) >= 2_000_000:
            raise ValueError("Complete inference student must contain fewer than 2M parameters")

    def encode(self, sequence):
        return self.encoder(sequence)[:, sequence.shape[-1] // 2]

    def forward(self, reference, alternate):
        if reference.shape != alternate.shape:
            raise ValueError("REF/ALT shapes must match")
        ref, alt = self.encode(reference), self.encode(alternate)
        features = torch.cat((ref, alt, alt - ref), dim=-1)
        logits = self.head((features - self.feature_mean) / self.feature_scale).squeeze(-1)
        return logits, features
