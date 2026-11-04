"""Real pretrained HyenaDNA forward/backward/save-reload smoke test.

This test uses a fixed synthetic DNA sequence and a randomly initialized head.
It never trains, calibrates, or reports a pathogenicity probability. Embeddings
are calculated by the real pinned pretrained encoder, not by a fake encoder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Keep all third-party caches inside this project and make the test offline.
PROJECT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(PROJECT / "work" / "huggingface")
os.environ["HF_MODULES_CACHE"] = str(PROJECT / "work" / "audited_hf_modules")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch

from evo2_mvp.hyenadna import (
    HyenaDNAStudent, PINNED_FILES, PRETRAINED_ID, PRETRAINED_REVISION,
    TOKEN_POLICY, file_digest, initialize_pretrained_student,
)
from evo2_mvp.models import one_hot
from evo2_mvp.utils import save_json


def run(pretrained_dir, window, output, device="cpu", threads=2, compare_official=False):
    if not 3 <= window <= 1024:
        raise ValueError("window must be in 3..1024")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    np.random.seed(42)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but this PyTorch runtime has no CUDA device")
    device = torch.device(device)
    pretrained_dir = Path(pretrained_dir).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, model_kwargs, provenance = initialize_pretrained_student(
        {"pretrained_dir": str(pretrained_dir), "pooling": "variant_center"}, window)
    model.to(device).eval()
    # The initializer metadata describes the training architecture; this smoke
    # test performs no optimizer step and explicitly overrides that run status.
    provenance["encoder_finetuned"] = False
    provenance["head_trained"] = False
    reference = ("ACGT" * ((window + 3) // 4))[:window]
    center = window // 2
    alt_base = {"A": "C", "C": "G", "G": "T", "T": "A"}[reference[center]]
    alternate = reference[:center] + alt_base + reference[center + 1:]
    ref_input = one_hot([reference]).to(device)
    alt_input = one_hot([alternate]).to(device)
    ref_ids = model.token_ids(ref_input)
    alt_ids = model.token_ids(alt_input)
    expected_ref_ids = [TOKEN_POLICY["base_to_id"][base] for base in reference]
    expected_alt_ids = [TOKEN_POLICY["base_to_id"][base] for base in alternate]
    assert ref_ids.tolist() == [expected_ref_ids]
    assert alt_ids.tolist() == [expected_alt_ids]
    assert torch.nonzero(ref_ids != alt_ids).tolist() == [[0, center]]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits, fused = model(ref_input, alt_input, variant_index=center)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_elapsed = time.perf_counter() - start
    with torch.inference_mode():
        ref_hidden = model.encoder(ref_ids, return_dict=True).last_hidden_state
        alt_hidden = model.encoder(alt_ids, return_dict=True).last_hidden_state
    ref_embedding = ref_hidden[:, center, :]
    alt_embedding = alt_hidden[:, center, :]
    delta = alt_embedding - ref_embedding
    assert tuple(ref_hidden.shape) == tuple(alt_hidden.shape) == (1, window, 256)
    assert tuple(ref_embedding.shape) == tuple(alt_embedding.shape) == (1, 256)
    assert tuple(fused.shape) == (1, 768)
    assert torch.equal(fused, torch.cat((ref_embedding, alt_embedding, delta), dim=1))
    assert all(torch.isfinite(value).all() for value in (ref_hidden, alt_hidden, fused, logits))
    assert float(delta.norm()) > 0, "Real pretrained REF/ALT representation difference is zero"

    arrays = {"embedding_ref": ref_embedding[0], "embedding_alt": alt_embedding[0],
              "embedding_delta": delta[0], "variant_features": fused[0],
              "token_hidden_ref": ref_hidden[0], "token_hidden_alt": alt_hidden[0]}
    for name, tensor in arrays.items():
        value = tensor.detach().cpu().numpy()
        path = output / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        assert np.array_equal(np.load(path, allow_pickle=False), value)

    # Test that a real differentiable path reaches the pretrained backbone.
    # There are no labels and no optimizer: the model weights remain unchanged.
    model.zero_grad(set_to_none=True)
    structural_logits, structural_features = model(ref_input, alt_input, variant_index=center)
    structural_loss = structural_features.square().mean() + structural_logits.square().mean()
    structural_loss.backward()
    gradients = [parameter.grad for parameter in model.encoder.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    encoder_gradient_norm = float(torch.sqrt(sum(gradient.detach().double().square().sum() for gradient in gradients)))
    assert encoder_gradient_norm > 0
    model.zero_grad(set_to_none=True)
    checkpoint_path = output / "structural_smoke_untrained_head.pt"
    checkpoint = {"kind": "structural_smoke_untrained_head", "architecture": "hyenadna_d256",
                  "model_kwargs": model_kwargs, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "provenance": provenance, "clinical_use": "research_only",
                  "warning": "Random head. Structural smoke test only; never use this checkpoint for pathogenicity prediction."}
    torch.save(checkpoint, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["kind"] == "structural_smoke_untrained_head"
    reloaded = HyenaDNAStudent(**saved["model_kwargs"])
    reloaded.load_state_dict(saved["model_state"], strict=True)
    reloaded.to(device).eval()
    with torch.inference_mode():
        reload_logits, reload_features = reloaded(ref_input, alt_input, variant_index=center)
    assert torch.equal(logits, reload_logits)
    assert torch.equal(fused, reload_features)

    comparison = {"requested": compare_official, "status": "not_run"}
    if compare_official:
        # This opt-in path executes only the previously reviewed source files
        # after rechecking every fixed digest. It never fetches remote code.
        for filename, expected in PINNED_FILES.items():
            assert file_digest(pretrained_dir / filename) == expected
        from transformers import AutoModel, AutoTokenizer
        official = AutoModel.from_pretrained(str(pretrained_dir), local_files_only=True,
            trust_remote_code=True, use_safetensors=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(pretrained_dir), local_files_only=True,
            trust_remote_code=True)
        official_ids = tokenizer([reference, alternate], add_special_tokens=False,
            padding=False, return_tensors="pt")["input_ids"].to(device)
        assert torch.equal(official_ids, torch.cat((ref_ids, alt_ids), dim=0))
        # Match per-sequence batch dimensions to avoid unrelated GEMM batching
        # differences when checking strict bitwise numerical equivalence.
        with torch.inference_mode():
            official_ref = official(official_ids[:1], return_dict=True).last_hidden_state
            official_alt = official(official_ids[1:], return_dict=True).last_hidden_state
        torch.testing.assert_close(official_ref, ref_hidden, rtol=0, atol=0)
        torch.testing.assert_close(official_alt, alt_hidden, rtol=0, atol=0)
        comparison = {"requested": True, "status": "passed", "local_files_only": True,
                      "all_pinned_source_digests_rechecked": True, "token_ids_equal": True,
                      "ref_and_alt_hidden_bitwise_equal": True, "max_abs_difference": 0.0}

    result = {"phase": "hyenadna_real_pretrained_structural_smoke", "status": "passed",
        "repo_id": PRETRAINED_ID, "revision": PRETRAINED_REVISION,
        "window": window, "variant_index": center, "ref_base": reference[center], "alt_base": alt_base,
        "input_sequence_source": "fixed_synthetic_ACGT_context_no_patient_data",
        "reference_sequence": reference, "alternate_sequence": alternate,
        "pretrained_encoder_loaded": True, "encoder_finetuned": False, "head_status": "random_untrained_structural_test_only",
        "pathogenicity_score": None, "pathogenicity_performance": None,
        "clinical_use": "research_only", "seed": 42,
        "device": str(device), "torch_version": torch.__version__, "cpu_threads": threads,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "encoder_parameter_count": sum(p.numel() for p in model.encoder.parameters()),
        "embedding_ref_shape": list(ref_embedding[0].shape), "embedding_alt_shape": list(alt_embedding[0].shape),
        "embedding_delta_shape": list(delta[0].shape), "variant_features_shape": list(fused[0].shape),
        "ref_token_hidden_shape": list(ref_hidden.shape), "alt_token_hidden_shape": list(alt_hidden.shape),
        "token_mapping": TOKEN_POLICY, "center_token_ids": {"ref": int(ref_ids[0, center]), "alt": int(alt_ids[0, center])},
        "finite": True, "delta_l2_norm": float(delta.norm()), "encoder_backward_gradient_norm": encoder_gradient_norm,
        "encoder_backward_finite_nonzero": True, "optimizer_steps": 0,
        "save_reload_logits_bitwise_equal": True, "save_reload_features_bitwise_equal": True,
        "npy_save_reload_equal": True, "inference_elapsed_seconds": inference_elapsed,
        "timing_scope": "single_REF_ALT_forward_including_random_head_no_warmup_not_a_benchmark",
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "pretrained_provenance": provenance, "official_implementation_comparison": comparison,
        "checkpoint": str(checkpoint_path), "array_files": {name: str(output / (name + ".npy")) for name in arrays}}
    save_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-dir", default="work/hyenadna_d256_pretrained")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--output", default="outputs/hyenadna_phase0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--compare-official", action="store_true", help="Offline comparison to audited fixed-hash official model code")
    args = parser.parse_args()
    result = run(args.pretrained_dir, args.window, args.output, args.device, args.threads, args.compare_official)
    print(json.dumps({key: result[key] for key in ("status", "repo_id", "revision", "parameter_count",
        "encoder_parameter_count", "embedding_ref_shape", "variant_features_shape", "delta_l2_norm",
        "encoder_backward_finite_nonzero", "save_reload_features_bitwise_equal", "official_implementation_comparison")}, indent=2))
    print("Saved " + str(Path(args.output).resolve() / "result.json"))


if __name__ == "__main__":
    main()
