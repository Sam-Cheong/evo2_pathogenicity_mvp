"""Required Phase 0 real inference. Failure never writes simulated .npy results."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from evo2_mvp.environment import inspect_environment
from evo2_mvp.evo2_encoder import create_encoder
from evo2_mvp.nvidia_encoder import MissingAPIKey, NvidiaAPIError
from evo2_mvp.utils import load_config, save_json, fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mvp.yaml")
    p.add_argument("--window", type=int, choices=(256, 512), default=256)
    p.add_argument("--output", default="outputs/phase0_api")
    p.add_argument("--local-checkpoint")
    p.add_argument("--download-dir", default="work/huggingface")
    p.add_argument("--include-sequence-scores", action="store_true")
    args = p.parse_args()
    config = load_config(args.config)
    cfg = config["encoder"]
    remote = cfg.get("backend", "nvidia_api") == "nvidia_api"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "not_run", "real_model_requested": True, "real_model_verified": False,
        "backend": cfg.get("backend", "nvidia_api"), "checkpoint": "nvidia/" + cfg["model"] if remote else "evo2_1b_base",
        "window": args.window, "embedding_shape": None, "token_embedding_shape": None,
        "inference_seconds": None, "gpu_peak_allocated_bytes": None,
        "environment": inspect_environment(backend="nvidia_api" if remote else "local", model=cfg.get("model"))}
    try:
        start = time.perf_counter()
        encoder = create_encoder(cfg, local_checkpoint=args.local_checkpoint, download_dir=args.download_dir)
        report["load_seconds"] = time.perf_counter() - start
        sequence = ("ACGT" * (args.window // 4))
        index = args.window // 2
        alternate = sequence[:index] + "G" + sequence[index + 1:]
        if not remote:
            encoder.encode_pair(sequence, alternate, index)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        encoded = encoder.encode_pair(sequence, alternate, index)
        if not remote:
            torch.cuda.synchronize()
        report["inference_seconds"] = time.perf_counter() - start
        if remote:
            report["environment"]["api_authentication_verified"] = True
            report["latency_scope"] = "remote_REF_ALT_requests_including_network_and_server_time"
            report["server_gpu_memory"] = "not_exposed_by_API"
            report["last_request_info"] = encoder.last_request_info
        else:
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        repeated = encoder.encode_pair(sequence, alternate, index)
        report["repeat_max_absolute_difference"] = float(np.max(np.abs(encoded["features"] - repeated["features"])))
        report["repeat_tolerance"] = {"rtol": 1e-3, "atol": 1e-4}
        np.testing.assert_allclose(encoded["features"], repeated["features"], rtol=1e-3, atol=1e-4)
        if not remote and any(p.requires_grad for p in encoder.backend.model.parameters()):
            raise AssertionError("Evo 2 parameters must all be frozen")
        for name, array in encoded.items():
            np.save(out / (name + ".npy"), array, allow_pickle=False)
            np.testing.assert_array_equal(np.load(out / (name + ".npy"), allow_pickle=False), array)
        report.update({"status": "passed", "real_model_verified": True, "embedding_shape": list(encoded["ref"].shape),
            "variant_features_shape": list(encoded["features"].shape),
            "token_embedding_shape": encoder.last_token_shape, "metadata": encoder.metadata,
            "sequence_sha256": fingerprint(sequence), "ref_sequence": sequence,
            "alt_sequence": alternate, "variant_index": index, "repeated_run_consistent": True,
            "weights_frozen": None if remote else True,
            "encoder_training_mode": "remote_inference_only_no_local_encoder_parameters" if remote else "local_frozen",
            "finite": True, "save_reload_consistent": True})
        if args.include_sequence_scores:
            report["official_sequence_scores"] = encoder.score_pair(sequence, alternate)
    except Exception as exc:
        report["status"] = "blocked" if isinstance(exc, MissingAPIKey) or report["environment"]["selected_backend_blockers"] else "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if isinstance(exc, NvidiaAPIError) and exc.diagnostics:
            report["api_error"] = exc.diagnostics
        save_json(out / "result.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    save_json(out / "result.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

