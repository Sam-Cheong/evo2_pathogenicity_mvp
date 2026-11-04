"""Compare the same held-out test predictions, refusing mismatched lineages."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evo2_mvp.utils import save_json, sha256_file
from evo2_mvp.models import load_checkpoint


def validate_student_comparison(supervised, distilled):
    """Distinguish a controlled KD experiment from historical score tables."""
    architecture = supervised.get("architecture", "cnn")
    if architecture != distilled.get("architecture", "cnn"):
        raise ValueError("Student architectures differ; this is not a controlled distillation comparison")
    for field in ("model_kwargs", "seed", "window", "config"):
        if supervised.get(field) != distilled.get(field):
            raise ValueError("Student comparison configuration mismatch: " + field)
    left, right = supervised.get("initialization"), distilled.get("initialization")
    if not left and not right and architecture == "cnn":
        return {"architecture": architecture, "matched": False, "status": "unverified_legacy_cnn",
                "note": "Legacy CNN checkpoints lack complete initial-state fingerprints; scores are descriptive, not a verified initialization-controlled KD comparison."}
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Student comparison requires initialization provenance for both checkpoints")
    for initialization in (left, right):
        digest = initialization.get("initial_model_state_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("Student initialization is missing its complete initial-model SHA256")
    if left != right:
        raise ValueError("Student initialization revision/weight/config/tokenizer/state fingerprints differ")
    left_protocol, right_protocol = supervised.get("initialization_protocol"), distilled.get("initialization_protocol")
    if not isinstance(left_protocol, dict) or left_protocol != right_protocol:
        raise ValueError("Student initialization protocols differ or are missing")
    if supervised.get("training_settings") != distilled.get("training_settings"):
        raise ValueError("Student optimization settings differ")
    return {"architecture": architecture, "matched": True, "status": "verified_identical_initialization",
            "initial_model_state_sha256": left["initial_model_state_sha256"],
            "initialization": left, "initialization_protocol": left_protocol}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, help="Teacher experiment output directory")
    p.add_argument("--supervised", required=True, help="Supervised student experiment directory")
    p.add_argument("--distilled", required=True, help="Distilled student experiment directory")
    p.add_argument("--teacher-e2e", help="Optional inference --benchmark-output JSON")
    p.add_argument("--supervised-e2e")
    p.add_argument("--distilled-e2e")
    p.add_argument("--output", default="outputs/comparison.json")
    args = p.parse_args(argv)
    reports, signatures, provenance, checkpoints = {}, [], [], {}
    teacher_hash = sha256_file(Path(args.teacher) / "best_teacher.pt")
    for name in ("teacher", "supervised", "distilled"):
        directory = Path(getattr(args, name))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        checkpoint_path = directory / ("best_teacher.pt" if name == "teacher" else "best_student.pt")
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoints[name] = checkpoint
        expected_kind = "teacher" if name == "teacher" else "student"
        if checkpoint["kind"] != expected_kind:
            raise ValueError("Comparison model kind mismatch: " + name)
        if name != "teacher" and (checkpoint["mode"] != ("supervised" if name == "supervised" else "distill")
                or checkpoint["teacher_sha256"] != teacher_hash):
            raise ValueError("Student training mode/teacher lineage mismatch: " + name)
        metadata = json.loads((directory / "prediction_metadata.json").read_text(encoding="utf-8"))
        if metadata["checkpoint_sha256"] != checkpoint_hash or metadata["predictions_sha256"] != sha256_file(directory / "predictions.csv"):
            raise ValueError("Prediction provenance checksum mismatch: " + name)
        if checkpoint["data_provenance"] != metrics["provenance"]:
            raise ValueError("Metrics data provenance mismatch: " + name)
        if metrics["provenance"].get("synthetic"):
            raise ValueError("Synthetic smoke metrics cannot be reported as real model comparison")
        with open(directory / "predictions.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        signatures.append(sorted((r["variant_id"], r["label"], r["split"]) for r in rows))
        provenance.append(metrics["provenance"])
        reports[name] = {"test": metrics["test"],
            "component_benchmark": json.loads((directory / "benchmark.json").read_text(encoding="utf-8")),
            "full_prediction_benchmark": None}
        e2e = getattr(args, name + "_e2e")
        if e2e:
            reports[name]["full_prediction_benchmark"] = json.loads(Path(e2e).read_text(encoding="utf-8"))
            benchmark = reports[name]["full_prediction_benchmark"]
            if benchmark.get("model_hashes", {}).get(expected_kind) != checkpoint_hash:
                raise ValueError("Full benchmark belongs to a different checkpoint: " + name)
    if not all(s == signatures[0] for s in signatures) or not all(p == provenance[0] for p in provenance):
        raise ValueError("Models must have exactly the same dataset, labels, and held-out splits")
    if len({c["split_fingerprint"] for c in checkpoints.values()}) != 1:
        raise ValueError("Checkpoint split identities differ")
    initialization_comparison = validate_student_comparison(checkpoints["supervised"], checkpoints["distilled"])
    full = [reports[name]["full_prediction_benchmark"] for name in reports]
    comparable = all(item is not None for item in full)
    if comparable:
        comparable = len({(x["variant_id"], x["window"], x["scope"], x["device"]) for x in full}) == 1
    output = {"models": reports, "same_independent_test_set": True,
        "student_initialization_comparison": initialization_comparison,
        "controlled_distillation_comparison": initialization_comparison["matched"],
        "latency_comparable": comparable,
        "latency_note": "Cached teacher head latency excludes Evo 2; use full warm prediction benchmarks for comparable inference cost.",
        "clinical_use": "research_only"}
    save_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

