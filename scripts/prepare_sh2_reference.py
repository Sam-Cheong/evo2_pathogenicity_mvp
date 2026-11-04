"""Pinned Savanna reference functions, loaded without CUDA/distributed imports.

Only original function/method AST nodes are compiled. Mathematical bodies remain
unchanged; JIT/torch.compile decorators are omitted. No model weights or API keys
are required. The full Savanna package is deliberately not imported.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
from urllib.request import Request, urlopen

COMMIT = "7b819b8609ce0b29deff79358a294b59d8299d07"
REPOSITORY = "https://github.com/Zymrael/savanna"
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "work" / "sh2_reference"
FILES = {
    "LICENSE": "4a3cf852673ee5b5cb90aacf640c99506951df3d4fae2023bb1d40da1e078e44",
    "README.md": "49b6ec859d20af48c21973061bc885ccd9825ee1cfa97e25f7a201c343ba504e",
    "savanna/ops/fftconv.py": "15e0ae59e8bf90b3beeaf0ad976975560b5aab49611169e940a3989b6ef52780",
    "savanna/model/operators/hyena/hyena.py": "cd1e1ded215ce27799bc03f66127f3189d064b1dced679c042a02811ce6e8e2c",
    "savanna/model/operators/hyena/parametrization/explicit.py": "2fb7b780ab2d61aa72e3561bd6eb2608c10d9a279e22d2269dd045cea9771978",
    "savanna/model/operators/hyena/parametrization/implicit_modal.py": "5f777d5af99f33c27c0350979a1ce9b7bfb54f6401e83cd3e41e07712cf56159",
    "savanna/model/operators/local/norms.py": "7f6faf11d628eccb0903ac3d6fd7a81c8c27e624ca8911b5c83032fcefaa1561",
    "savanna/model/operators/local/base.py": "7568d71d70d120eab1fda14c8d7848407ebb364551e13ed3c2925bf3d46d1707",
    "savanna/model/operators/positional_embeddings.py": "f8a9d084117b5a6b4375ed40850837c652b727795631a04be40018a15df32cd0",
    "savanna/model/block.py": "efd5be8db64af735346b39156c5a0690b4639ad5e50b7a9207e9000723c5ff0f",
    "configs/model/evo2/40b_test_config.yml": "fa1eddcbb1d6bd71cffd6b54c5553d19d85b9120acd7a5ef617bd065f35b2d98",
    "configs/model/evo2/7b_13h_8m_8s_3a_cascade15.yml": "d4a236a0733b1fd68bcb6854e9970aa52278ef1dcf890085c41a25f2c4b952be",
}
SYMBOLS = {
    "fftconv_func": ("savanna/ops/fftconv.py", None, "fftconv_func"),
    "ExplicitSingleDecayFilter.filter": ("savanna/model/operators/hyena/parametrization/explicit.py", "ExplicitSingleDecayFilter", "filter"),
    "ImplicitModalFilter.compute_filter": ("savanna/model/operators/hyena/parametrization/implicit_modal.py", "ImplicitModalFilter", "compute_filter"),
    "ParallelCausalDepthwiseConv1d.forward": ("savanna/model/operators/hyena/hyena.py", "ParallelCausalDepthwiseConv1d", "forward"),
    "ParallelShortHyenaOperator.forward": ("savanna/model/operators/hyena/hyena.py", "ParallelShortHyenaOperator", "forward"),
    "RMSNorm.forward": ("savanna/model/operators/local/norms.py", "RMSNorm", "forward"),
    "ParallelGLU.forward_func": ("savanna/model/operators/local/base.py", "ParallelGLU", "forward_func"),
    "rotate_half": ("savanna/model/operators/positional_embeddings.py", None, "rotate_half"),
    "apply_rotary_pos_emb_torch": ("savanna/model/operators/positional_embeddings.py", None, "apply_rotary_pos_emb_torch"),
}


def source_url(relative: str) -> str:
    return f"https://raw.githubusercontent.com/Zymrael/savanna/{COMMIT}/{relative}"


def _verify(relative: str, content: bytes) -> None:
    actual = hashlib.sha256(content).hexdigest()
    if actual != FILES[relative]:
        raise ValueError(f"Official source SHA256 mismatch: {relative}: {actual}")


def prepare(reference_dir: Path | str = DEFAULT_REFERENCE_DIR, download: bool = False) -> dict:
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for relative, digest in FILES.items():
        path = reference_dir / relative
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; run with --download once")
            request = Request(source_url(relative), headers={"User-Agent": "evo2-mvp-official-source-audit"})
            with urlopen(request, timeout=60) as response:
                content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("Reference file unexpectedly exceeds 2 MB")
            _verify(relative, content)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        _verify(relative, content)
        records.append({"path": relative, "url": source_url(relative), "sha256": digest, "bytes": len(content)})
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "license": "Apache-2.0", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "files": records, "weights_downloaded": False}
    (reference_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_official_function(name: str, reference_dir: Path | str = DEFAULT_REFERENCE_DIR):
    """Return an original function or unbound method; imports are not executed.

    Supported names are listed in SYMBOLS. For methods pass an object containing
    the original method's required state as the first argument. CUDA/distributed
    initializers are never called. Only single-process CPU fallback paths should
    be selected for the two convolution methods.
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    relative, class_name, function_name = SYMBOLS[name]
    content = (Path(reference_dir) / relative).read_bytes()
    _verify(relative, content)
    tree = ast.parse(content.decode("utf-8"), filename=relative)
    container = tree.body
    if class_name:
        container = next(node for node in container if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = copy.deepcopy(next(node for node in container if isinstance(node, ast.FunctionDef) and node.name == function_name))
    node.decorator_list = []  # Compile/JIT wrappers only; original function body is untouched.
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = {"torch": torch, "nn": torch.nn, "F": F, "rearrange": rearrange, "math": math,
           "mpu": SimpleNamespace(get_sequence_parallel_world_size=lambda: 1)}
    if name == "apply_rotary_pos_emb_torch":
        env["rotate_half"] = load_official_function("rotate_half", reference_dir)
    exec(compile(module, f"{REPOSITORY}/blob/{COMMIT}/{relative}", "exec"), env)
    return env[function_name]


def load_official_method(class_name: str, method_name: str, reference_dir: Path | str = DEFAULT_REFERENCE_DIR):
    return load_official_function(f"{class_name}.{method_name}", reference_dir)


def run_self_checks(reference_dir: Path | str = DEFAULT_REFERENCE_DIR) -> dict:
    """Compare original reference bodies to independent CPU formulas and gradients."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(7)
    results = {}

    def compare(name, official, independent, variables, atol=1e-10, rtol=1e-9):
        actual = official(*variables)
        expected = independent(*variables)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        probe = torch.linspace(0.1, 0.9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        grad_actual = torch.autograd.grad((actual * probe).sum(), variables)
        grad_expected = torch.autograd.grad((expected * probe).sum(), variables)
        for ga, ge in zip(grad_actual, grad_expected):
            torch.testing.assert_close(ga, ge, atol=atol * 20, rtol=rtol * 20)
        results[name] = {"passed": True, "output_max_abs_difference": (actual - expected).abs().max().item(), "gradient_max_abs_difference": max((ga - ge).abs().max().item() for ga, ge in zip(grad_actual, grad_expected))}

    def tensor(*shape, dtype=torch.float64):
        return torch.randn(*shape, dtype=dtype, requires_grad=True)

    fft = load_official_function("fftconv_func", reference_dir)

    def causal(u, h, d):
        kernel = h.squeeze(0) if h.ndim == 3 else h
        return F.conv1d(F.pad(u, (kernel.shape[-1] - 1, 0)), kernel.flip(-1)[:, None], groups=u.shape[1]) + u * d[None, :, None]

    for length in (5, 17):
        variables = (tensor(2, 4, 17), tensor(4, length), tensor(4))
        compare(f"fftconv_filter_length_{length}", lambda u, h, d: fft(u, h, d, None, gelu=False), causal, variables)

    conv = load_official_method("ParallelCausalDepthwiseConv1d", "forward", reference_dir)
    se = load_official_method("ParallelShortHyenaOperator", "forward", reference_dir)

    def official_se(x1, x2, v, weight):
        conv_state = SimpleNamespace(short_conv_weight=weight, kernel_size=5, group_dim=1, use_cp_hyena=False, use_fast_causal_conv=False, conv_groups=4)
        state = SimpleNamespace(use_custom_hyena_mlp_kernel=False, use_custom_hyena_short_kernel=False, use_cgcg_mlp=False, use_cgcg_short=False, pregate=True, postgate=True, short_conv=lambda u: conv(conv_state, u))
        return se(state, x1.permute(0, 2, 1)[..., None], x2.permute(0, 2, 1)[..., None], v.permute(0, 2, 1)[..., None]).permute(0, 2, 1)

    compare("hyena_se_gates_and_original_short_conv", official_se,
            lambda x1, x2, v, w: x1 * F.conv1d(F.pad(x2 * v, (4, 0)), w, groups=4),
            (tensor(2, 4, 17), tensor(2, 4, 17), tensor(2, 4, 17), tensor(4, 1, 5)))

    mr = load_official_method("ExplicitSingleDecayFilter", "filter", reference_dir)
    rates = torch.logspace(-2, 2, 4, dtype=torch.float64)[:, None]
    times = torch.linspace(0, 1, 9, dtype=torch.float64)[None]
    compare("hyena_mr_explicit_filter", lambda a: mr(SimpleNamespace(h=a, decay=torch.exp(-rates * times)), 7),
            lambda a: a[:, :7] * torch.exp(-rates * times[:, :7]), (tensor(4, 9),))

    li = load_official_method("ImplicitModalFilter", "compute_filter", reference_dir)
    times_li = torch.arange(17, dtype=torch.float32)[None, None]

    def independent_li(p, gamma, residues):
        pole = torch.exp(-torch.exp(p) * torch.exp(gamma))
        power = torch.ones_like(pole)
        values = []
        for _ in range(17):
            values.append((residues * power).sum(-1))
            power = power * pole
        return torch.stack(values, dim=-1)[None]

    variables = (tensor(4, 3, dtype=torch.float32), tensor(4, 3, dtype=torch.float32), tensor(4, 3, dtype=torch.float32))
    compare("hyena_li_original_modal_filter", lambda p, g, r: li(SimpleNamespace(p=p, gamma=g, R=r), 17, times_li)[0], independent_li, variables, atol=3e-6, rtol=3e-5)
    compare("hyena_mr_li_double_gate_skip", lambda x1, x2, v, h, d: x1 * fft(x2 * v, h, d, None, gelu=False),
            lambda x1, x2, v, h, d: x1 * causal(x2 * v, h, d),
            (tensor(2, 4, 17), tensor(2, 4, 17), tensor(2, 4, 17), tensor(4, 7), tensor(4)))

    norm = load_official_method("RMSNorm", "forward", reference_dir)
    compare("savanna_rmsnorm_epsilon_outside_sqrt", lambda x, w: norm(SimpleNamespace(p=-1, d=4, eps=1e-6, scale=w, bias=False), x),
            lambda x, w: w * x / (torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)) + 1e-6), (tensor(2, 17, 4), tensor(4)))
    report = {"commit": COMMIT, "device": "cpu", "torch_version": torch.__version__, "mathematical_bodies_unmodified": True, "training_or_model_api_calls": False, "checks": results}
    (Path(reference_dir) / "reference_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--download", action="store_true", help="Download missing pinned small public source/license files")
    parser.add_argument("--self-check", action="store_true", help="Run CPU value and gradient reference checks")
    args = parser.parse_args()
    manifest = prepare(args.reference_dir, download=args.download)
    report = run_self_checks(args.reference_dir) if args.self_check else {"commit": manifest["commit"], "verified_files": len(manifest["files"])}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
