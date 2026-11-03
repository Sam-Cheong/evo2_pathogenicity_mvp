from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone


REPO = "arcinstitute/evo2_1b_base"


def inspect_environment(check_network=False, backend="nvidia_api", model=None):
    import torch
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "system": platform.system(),
        "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(), "gpus": [],
        "packages": {}, "official_model": REPO,
        "official_requirements_source": "https://github.com/ArcInstitute/evo2#requirements",
    }
    for name in ("evo2", "vtx", "transformer-engine", "flash-attn", "numpy", "pyfaidx"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            report["gpus"].append({"index": i, "name": p.name,
                "compute_capability": list(torch.cuda.get_device_capability(i)),
                "total_bytes": total, "free_bytes": free})
    if shutil.which("nvidia-smi"):
        try:
            report["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia_smi_error"] = str(exc)
    if shutil.which("nvcc"):
        try:
            report["nvcc"] = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvcc_error"] = str(exc)
    blockers = []
    if platform.system() != "Linux":
        blockers.append("Official runtime requires Linux; WSL2 has limited support. Native Windows is unsupported.")
    if not ((3, 11) <= sys.version_info[:2] <= (3, 12)):
        blockers.append("Official Evo 2 requires Python 3.11 or 3.12.")
    if not torch.cuda.is_available():
        blockers.append("CUDA PyTorch is unavailable.")
    elif any(g["compute_capability"][0] < 9 for g in report["gpus"]):
        blockers.append("Evo 2 1B requires Hopper-class FP8/Transformer Engine; this GPU is below compute capability 9.0.")
    if torch.backends.cudnn.version() is None or torch.backends.cudnn.version() < 90300:
        blockers.append("Official Evo 2 runtime requires cuDNN 9.3 or newer.")
    for module, label in (("evo2", "evo2"), ("transformer_engine", "Transformer Engine"), ("flash_attn", "Flash Attention")):
        if importlib.util.find_spec(module) is None:
            blockers.append(label + " is not installed.")
    report["official_1b_preflight_blockers"] = blockers
    report["ready_for_official_1b_attempt"] = not blockers
    report["measured_1b_peak_memory_bytes"] = None
    report["note"] = "Preflight readiness does not establish successful inference or a VRAM requirement."
    report["selected_backend"] = backend
    if backend == "nvidia_api":
        report["official_model"] = "nvidia/" + (model or "arc/evo2-40b")
        report["official_requirements_source"] = "https://docs.nvidia.com/nim/bionemo/evo2/endpoints"
        from .nvidia_encoder import load_api_key, MissingAPIKey
        try:
            load_api_key()
            present = True
        except MissingAPIKey:
            present = False
        report["api_key_present"] = present
        report["selected_backend_blockers"] = [] if present else ["NVIDIA_API_KEY is missing; set the project .env or process environment."]
        report["ready_for_selected_backend_attempt"] = present
        report["api_authentication_verified"] = False
        report["local_gpu_required"] = False
        report["local_evo2_installation_required"] = False
        for field in ("official_1b_preflight_blockers", "ready_for_official_1b_attempt", "measured_1b_peak_memory_bytes"):
            report.pop(field, None)
        report["note"] = "Hosted Evo 2 requires no local CUDA/Evo 2 weights. Key presence is not successful authentication; run Phase 0."
        if check_network:
            report["api_network_check"] = "not_invoked; Phase 0 performs authenticated forward requests"
    else:
        report["selected_backend_blockers"] = blockers
        report["ready_for_selected_backend_attempt"] = not blockers
    if check_network and backend == "local":
        url = "https://huggingface.co/api/models/" + REPO
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "evo2-pathogenicity-mvp/0.1"})
            with urllib.request.urlopen(request, timeout=15) as response:
                metadata = json.load(response)
            report["weight_access"] = {"status": "metadata_accessible", "sha": metadata.get("sha"),
                "gated": metadata.get("gated"), "private": metadata.get("private"),
                "license": metadata.get("cardData", {}).get("license")}
        except Exception as exc:
            report["weight_access"] = {"status": "metadata_request_failed", "error": str(exc)}
    return report


def require_official_runtime():
    report = inspect_environment(backend="local")
    if report["official_1b_preflight_blockers"]:
        raise RuntimeError("Official Evo 2 1B preflight failed: " + " ".join(report["official_1b_preflight_blockers"]))
    return report

