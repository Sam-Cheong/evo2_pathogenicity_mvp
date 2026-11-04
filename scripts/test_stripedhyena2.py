"""Actual CPU forward/backward/reload test; no pathogenicity score is reported."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evo2_mvp.models import one_hot
from evo2_mvp.stripedhyena2 import StripedHyena2Student, REFERENCE_REVISION
from evo2_mvp.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/sh2_two_stage_20261104/phase0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {"d_model": 192, "ffn_dim": 384, "num_heads": 6, "window": 256, "head_kind": "linear", "dropout": .1}
    model = StripedHyena2Student(**kwargs).eval()
    ref = "ACGT" * 64
    alt = ref[:128] + "C" + ref[129:]
    reference, alternate = one_hot([ref]), one_hot([alt])
    started = time.perf_counter()
    with torch.no_grad():
        embedding_ref, embedding_alt = model.encode(reference), model.encode(alternate)
        logits, features = model(reference, alternate)
    elapsed = time.perf_counter() - started
    if not torch.isfinite(features).all() or torch.equal(embedding_ref, embedding_alt):
        raise RuntimeError("Invalid/nonfinite or insensitive SNV representation")
    _, backward_features = model(reference, alternate)
    (backward_features * torch.linspace(-1, 1, 576)).square().mean().backward()
    gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or sum(float(g.abs().sum()) for g in gradients) == 0:
        raise RuntimeError("Invalid encoder gradients")
    checkpoint = {"kind": "structural_smoke_untrained_SH2", "model_kwargs": kwargs,
                  "model_state": model.state_dict(), "pathogenicity_score": None}
    path = output / "structural_untrained.pt"
    torch.save(checkpoint, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = StripedHyena2Student(**saved["model_kwargs"]).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    with torch.no_grad():
        reloaded_logits, reloaded_features = restored(reference, alternate)
    if not torch.equal(logits, reloaded_logits) or not torch.equal(features, reloaded_features):
        raise RuntimeError("Saved model reload differs")
    arrays = {"embedding_ref": embedding_ref[0].numpy(), "embedding_alt": embedding_alt[0].numpy(),
              "embedding_delta": (embedding_alt - embedding_ref)[0].numpy(), "features": features[0].numpy()}
    for name, values in arrays.items():
        np.save(output / f"{name}.npy", values)
        if not np.array_equal(np.load(output / f"{name}.npy", allow_pickle=False), values):
            raise RuntimeError("NPY reload mismatch")
    report = {"status": "passed", "architecture": "custom_small_StripedHyena2_reference_variant",
        "reference_revision": REFERENCE_REVISION, "initialization": "seeded_random_not_pretrained",
        "teacher_cache_used_in_structural_test": False, "pathogenicity_score": None,
        "window": 256, "variant_index": 128, "embedding_shape": [192], "fused_shape": [576],
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "linear_student_parameters": sum(p.numel() for p in model.parameters()),
        "mlp_student_parameters": sum(p.numel() for p in StripedHyena2Student(head_kind="mlp").parameters()),
        "finite": True, "backward_finite_nonzero": True, "save_reload_bitwise_equal": True,
        "npy_reload_equal": True, "forward_seconds": elapsed, "device": "cpu", "cpu_threads": args.threads,
        "new_evo2_api_calls": 0, "clinical_use": "research_only"}
    save_json(output / "result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
