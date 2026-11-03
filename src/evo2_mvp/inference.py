"""Research-only SNV scores. Student-only inference never imports Evo 2."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .calibration import predictive_entropy
from .models import build_student, build_teacher, load_checkpoint, one_hot
from .sequence import ReferenceGenome, Variant
from .utils import save_json, sha256_file
from .provenance import validate_evo2_metadata


def check_research_checkpoint(checkpoint):
    provenance = checkpoint.get("data_provenance", {})
    if provenance.get("synthetic") is not False or checkpoint.get("n_epochs_run", 0) < 1:
        raise ValueError("Prediction requires a trained real-data checkpoint; synthetic/random smoke models are rejected.")
    metadata = checkpoint.get("cache_metadata", {})
    try:
        validate_evo2_metadata(metadata)
    except ValueError as exc:
        raise ValueError("Checkpoint lineage does not establish a real Evo 2 teacher experiment: " + str(exc)) from None
    if metadata.get("genome_build") != "GRCh38":
        raise ValueError("Checkpoint is not a GRCh38 research model.")


class Predictor:
    def __init__(self, fasta, teacher_path=None, student_path=None, device="cpu",
                 local_checkpoint=None, download_dir="work/huggingface", calibration_path=None):
        if not teacher_path and not student_path:
            raise ValueError("Supply at least --teacher or --student")
        self.reference = ReferenceGenome(fasta)
        self.reference_hash = sha256_file(fasta)
        self.device = torch.device(device)
        self.teacher = self.student = self.encoder = None
        self.teacher_checkpoint = self.student_checkpoint = None
        self.model_hashes = {}
        self.calibration = None
        self.window = None
        for name, path in (("teacher", teacher_path), ("student", student_path)):
            if path is None:
                continue
            checkpoint = load_checkpoint(path)
            check_research_checkpoint(checkpoint)
            if checkpoint["kind"] != name:
                raise ValueError(f"--{name} received a different model kind")
            metadata = checkpoint["cache_metadata"]
            if metadata["reference_sha256"] != self.reference_hash:
                raise ValueError("FASTA SHA256 differs from the training reference; use the same GRCh38 file.")
            window = int(metadata["window"])
            if self.window is not None and window != self.window:
                raise ValueError("Teacher and student windows differ")
            self.window = window
            self.model_hashes[name] = sha256_file(path)
            setattr(self, name + "_checkpoint", checkpoint)
            setattr(self, name, (build_teacher if name == "teacher" else build_student)(checkpoint, self.device))
        if self.teacher and self.student:
            if self.student_checkpoint["teacher_sha256"] != self.model_hashes["teacher"]:
                raise ValueError("Student was distilled from a different teacher")
            if self.teacher_checkpoint["split_fingerprint"] != self.student_checkpoint["split_fingerprint"]:
                raise ValueError("Teacher and student experimental splits differ")
        if self.teacher:
            # Deliberately lazy: the student-only path needs neither Evo 2 nor its weights/runtime.
            from .evo2_encoder import create_encoder, encoder_config_from_metadata
            m = self.teacher_checkpoint["cache_metadata"]
            self.encoder = create_encoder(encoder_config_from_metadata(m),
                local_checkpoint=local_checkpoint, download_dir=download_dir)
            observed = self.encoder.metadata
            if observed["feature_identity"] != m["feature_identity"]:
                raise ValueError("Inference feature protocol differs from the trained cache")
        if calibration_path:
            self.calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            kind = self.calibration["model_kind"]
            checkpoint = getattr(self, kind + "_checkpoint", None)
            if checkpoint is None or self.calibration["checkpoint_sha256"] != self.model_hashes.get(kind):
                raise ValueError("Calibration belongs to a different checkpoint")
            if (self.calibration["data_provenance"] != checkpoint["data_provenance"] or
                    self.calibration["split_fingerprint"] != checkpoint["split_fingerprint"] or self.calibration["fit_split"] != "val"):
                raise ValueError("Calibration experiment provenance does not match")
            t = self.calibration["temperature"]
            if not np.isfinite(t) or t <= 0:
                raise ValueError("Invalid calibration temperature")

    @torch.inference_mode()
    def predict(self, variant):
        context = self.reference.context(variant, self.window)
        result = {"variant_id": variant.id, "genome_build": "GRCh38",
            "teacher_score": None, "student_score": None, "student_predictive_entropy": None,
            "ensemble_std": None, "calibrated_score": None,
            "calibration_status": "not_calibrated", "clinical_use": "research_only"}
        if self.teacher:
            encoded = self.encoder.encode_pair(context.ref_sequence, context.alt_sequence, context.variant_index)
            tensor = torch.from_numpy(encoded["features"]).unsqueeze(0).to(self.device)
            teacher_logit = self.teacher(tensor)
            result["teacher_score"] = float(teacher_logit.sigmoid().item())
        if self.student:
            logits, _ = self.student(one_hot([context.ref_sequence]).to(self.device), one_hot([context.alt_sequence]).to(self.device))
            result["student_score"] = float(logits.sigmoid().item())
            result["student_predictive_entropy"] = predictive_entropy(result["student_score"])
        if self.calibration:
            logit = teacher_logit if self.calibration["model_kind"] == "teacher" else logits
            result["calibrated_score"] = float((logit / self.calibration["temperature"]).sigmoid().item())
            result["calibration_status"] = self.calibration["calibration_status"]
            result["calibrated_model"] = self.calibration["model_kind"]
        for key in ("teacher_score", "student_score", "student_predictive_entropy", "calibrated_score"):
            if result[key] is not None and not np.isfinite(result[key]):
                raise FloatingPointError("Nonfinite output: " + key)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--pos", required=True, type=int)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--gene", default="")
    p.add_argument("--teacher")
    p.add_argument("--student")
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-checkpoint")
    p.add_argument("--download-dir", default="work/huggingface")
    p.add_argument("--output")
    p.add_argument("--calibration", help="Validation-only temperature artifact tied to this checkpoint")
    p.add_argument("--benchmark-output", help="Measure complete warm prediction incl. FASTA/one-hot/encoder/head")
    args = p.parse_args()
    started = time.perf_counter()
    predictor = Predictor(args.fasta, args.teacher, args.student, args.device, args.local_checkpoint, args.download_dir, args.calibration)
    startup_seconds = time.perf_counter() - started
    variant = Variant(args.chrom, args.pos, args.ref, args.alt, args.gene)
    result = predictor.predict(variant)
    if args.output:
        save_json(args.output, result)
    if args.benchmark_output:
        from .metrics import benchmark_callable
        if args.teacher and args.student:
            raise ValueError("Benchmark one model per invocation so model timings are separable")
        benchmark = benchmark_callable(lambda: predictor.predict(variant), torch.device(args.device))
        benchmark.update({"scope": "warm_full_SNV_prediction_FASTA_encoder_head_and_uncertainty",
            "model": "teacher" if args.teacher else "student", "model_hashes": predictor.model_hashes,
            "variant_id": variant.id, "window": predictor.window, "startup_seconds": startup_seconds})
        save_json(args.benchmark_output, benchmark)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

