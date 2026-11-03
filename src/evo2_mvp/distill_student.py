"""Supervised and distilled CNN training after a completed teacher experiment.

Teacher information enters training only. Saved student checkpoints contain no
Evo 2 model or teacher projection, and inference imports no Evo 2 package.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cache import FeatureCache
from .calibration import predictive_entropy
from .metrics import (benchmark_callable, binary_metrics, file_sha256, fingerprint, set_seed,
                      write_csv, write_json, write_prediction_metadata)
from .models import TinyStudent, build_student, build_teacher, load_checkpoint, one_hot
from .provenance import validate_evo2_metadata
from .sequence import ReferenceGenome
from .train_teacher_head import (check_cache, choose_device, experiment_data, feature_tensors,
                                 read_config, save_history, training_options)


def binary_kd_loss(student_logits, teacher_logits, temperature=2.0):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("KD temperature must be finite and positive")
    targets = torch.sigmoid(teacher_logits.detach() / temperature)
    return F.binary_cross_entropy_with_logits(student_logits / temperature, targets) * temperature**2


def sequence_tensors(variants, splits, fasta, window):
    reference = ReferenceGenome(fasta)
    by_id = {v.id: v for v in variants}
    result = {}
    try:
        for name in ("train", "val", "test"):
            contexts = [reference.context(by_id[key], window) for key in splits[name]]
            result[name] = (one_hot([c.ref_sequence for c in contexts]),
                            one_hot([c.alt_sequence for c in contexts]),
                            torch.tensor([by_id[key].label for key in splits[name]], dtype=torch.float32))
    finally:
        if hasattr(reference, "close"):
            reference.close()
    return result


def teacher_targets(teacher, features, device):
    targets = {}
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        for split, (values, _) in features.items():
            targets[split] = torch.cat([teacher(batch.to(device)).cpu() for batch in values.split(128)])
    return targets


def fit_student(model, sequences, features, teacher, config, device, mode="distill"):
    if mode not in ("supervised", "distill"):
        raise ValueError("Unknown student training mode")
    options = training_options(config)
    kd = {"alpha": 1.0, "beta": 0.1, "gamma": 0.5, "temperature": 2.0, "projection_dim": 128}
    kd.update(config.get("distillation", {}))
    if any(kd[name] < 0 for name in ("alpha", "beta", "gamma")) or kd["alpha"] <= 0:
        raise ValueError("Distillation weights must be nonnegative with alpha > 0")
    targets = teacher_targets(teacher, features, device) if mode == "distill" else None
    # Fixed seeded random teacher projection prevents a jointly trained projection
    # from collapsing both targets to zero. Normalization statistics are fitted
    # on the teacher's training set only and stored in its checkpoint.
    kd_dim = int(kd["projection_dim"])
    if kd_dim <= 0:
        raise ValueError("Projection dimension must be positive")
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    fixed_projection = torch.randn(features["train"][0].shape[1], kd_dim, generator=generator) / np.sqrt(kd_dim)
    feature_targets = {}
    if mode == "distill":
        mean, scale = teacher.input_mean.detach().cpu(), teacher.input_scale.detach().cpu()
        for split, (values, _) in features.items():
            feature_targets[split] = F.normalize(((values - mean) / scale) @ fixed_projection, p=2, dim=-1)
    projector = nn.Linear(model.feature_dim, kd_dim).to(device)
    model.to(device)
    parameters = list(model.parameters()) + (list(projector.parameters()) if mode == "distill" else [])
    optimizer = torch.optim.AdamW(parameters, lr=options["lr"], weight_decay=options["weight_decay"])
    ref, alt, labels = sequences["train"]
    loader = DataLoader(TensorDataset(ref, alt, labels, torch.arange(len(labels))),
                        batch_size=options["batch_size"], shuffle=True)
    history, best_state, best_projector = [], None, None
    best_validation, bad_epochs = float("inf"), 0
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        projector.train()
        sums = {name: 0.0 for name in ("loss", "supervised", "feature", "prediction_kd")}
        for batch_ref, batch_alt, batch_labels, indices in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, representation = model(batch_ref.to(device), batch_alt.to(device))
            supervised = F.binary_cross_entropy_with_logits(logits, batch_labels.to(device))
            feature, prediction = logits.new_zeros(()), logits.new_zeros(())
            if mode == "distill":
                feature = F.mse_loss(F.normalize(projector(representation), p=2, dim=-1),
                                     feature_targets["train"][indices].to(device))
                prediction = binary_kd_loss(logits, targets["train"][indices].to(device), kd["temperature"])
                loss = kd["alpha"] * supervised + kd["beta"] * feature + kd["gamma"] * prediction
            else:
                loss = supervised
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite student loss")
            loss.backward()
            optimizer.step()
            for name, item in (("loss", loss), ("supervised", supervised), ("feature", feature), ("prediction_kd", prediction)):
                sums[name] += item.item() * len(batch_labels)
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(r.to(device), a.to(device))[0].cpu()
                                    for r, a in zip(sequences["val"][0].split(128), sequences["val"][1].split(128))])
            validation_loss = F.binary_cross_entropy_with_logits(val_logits, sequences["val"][2]).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite student validation loss")
        row = {"epoch": epoch, "train_loss": sums["loss"] / len(labels), "val_loss": validation_loss,
               "train_supervised": sums["supervised"] / len(labels), "train_feature": sums["feature"] / len(labels),
               "train_prediction_kd": sums["prediction_kd"] / len(labels)}
        history.append(row)
        if validation_loss < best_validation - 1e-8:
            best_validation, bad_epochs = validation_loss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_projector = copy.deepcopy({k: v.detach().cpu() for k, v in projector.state_dict().items()})
        else:
            bad_epochs += 1
        if bad_epochs >= options["patience"]:
            break
    model.load_state_dict(best_state)
    training_artifact = {"fixed_teacher_projection": fixed_projection,
                         "student_projector_state": best_projector,
                         "distillation_config": kd,
                         "projection_protocol": "fixed_seeded_random_teacher_projection_L2_normalized;train_only_feature_scaler"}
    return model.eval(), history, training_artifact


def report_student(model, sequences, splits, output, device, provenance, checkpoint_path):
    metrics, rows = {}, []
    with torch.inference_mode():
        for name in ("train", "val", "test"):
            ref, alt, labels = sequences[name]
            logits = torch.cat([model(r.to(device), a.to(device))[0].cpu()
                                for r, a in zip(ref.split(128), alt.split(128))])
            scores = logits.sigmoid().numpy()
            metrics[name] = binary_metrics(labels.numpy(), scores)
            entropies = predictive_entropy(scores)
            rows.extend({"variant_id": key, "split": name, "label": int(label), "logit": float(logit),
                         "score": float(score), "predictive_entropy_nats": float(entropy)}
                        for key, label, logit, score, entropy in zip(splits[name], labels, logits.numpy(), scores, entropies))
        sample_ref, sample_alt = sequences["test"][0][:1].to(device), sequences["test"][1][:1].to(device)
        benchmark = benchmark_callable(lambda: model(sample_ref, sample_alt), device)
    benchmark["scope"] = "student_encoder_and_head_preencoded_one_hot_excludes_FASTA_and_one_hot_preprocessing"
    benchmark["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["provenance"] = provenance
    write_json(Path(output) / "metrics.json", metrics)
    write_csv(Path(output) / "predictions.csv", rows)
    write_prediction_metadata(output, rows, checkpoint_path, splits, provenance)
    write_json(Path(output) / "benchmark.json", benchmark)
    return metrics


def check_phase0(path, cache_metadata):
    if not path:
        raise ValueError("Run a successful real Evo 2 Phase 0 first and provide --phase0-report")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Phase 0 must identify the real encoder metadata")
    verified = report.get("real_model_verified", metadata.get("real_model_verified"))
    if report.get("status") not in {"passed", "success"} or verified is not True:
        raise ValueError("Phase 0 was not verified with the real Evo 2 model; student stage remains gated")
    validate_evo2_metadata(metadata, require_verified=True)
    validate_evo2_metadata(cache_metadata, require_verified=True)
    if metadata.get("feature_identity") != cache_metadata.get("feature_identity"):
        raise ValueError("Phase 0 encoder identity differs from the teacher cache; verify this exact backend/model/layer/run first")
    window = cache_metadata.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1 or report.get("window") != window:
        raise ValueError("Phase 0 sequence window differs from the teacher cache window")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["supervised", "distill"], default="distill")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--phase0-report")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--allow-synthetic-smoke-test", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)
    variants, splits, provenance = experiment_data(args.variants, args.splits, args.allow_synthetic_smoke_test)
    cache = FeatureCache(args.cache)
    check_cache(cache, provenance)
    teacher_checkpoint = load_checkpoint(args.teacher)
    if teacher_checkpoint["kind"] != "teacher":
        raise ValueError("A trained teacher classification checkpoint is required")
    if teacher_checkpoint["split_fingerprint"] != fingerprint(splits) or teacher_checkpoint["data_provenance"] != provenance:
        raise ValueError("Student/teacher must use exactly the same variants, labels, and split")
    if teacher_checkpoint["cache_metadata"] != cache.metadata:
        raise ValueError("Teacher and student cache metadata differ")
    if not provenance["synthetic"]:
        check_phase0(args.phase0_report, cache.metadata)
        if cache.metadata.get("reference_sha256") != file_sha256(args.fasta):
            raise ValueError("Student FASTA differs from the teacher cache reference")
    window = int(cache.metadata.get("window", config.get("window", 256)))
    sequences = sequence_tensors(variants, splits, args.fasta, window)
    features = feature_tensors(variants, splits, cache)
    teacher = build_teacher(teacher_checkpoint, device)
    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate)
        if checkpoint["split_fingerprint"] != fingerprint(splits) or checkpoint["data_provenance"] != provenance:
            raise ValueError("Evaluation dataset/split differs from student training")
        if checkpoint["teacher_sha256"] != file_sha256(args.teacher):
            raise ValueError("Student teacher lineage differs from the supplied experiment")
        model = build_student(checkpoint, device)
    else:
        model_kwargs = {key: config.get("student", {}).get(key, value)
                        for key, value in {"embedding_dim": 64, "channels": [32, 64, 96], "dropout": 0.2}.items()}
        model = TinyStudent(**model_kwargs)
        model, history, artifact = fit_student(model, sequences, features, teacher, config, device, args.mode)
        checkpoint = {"kind": "student", "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                      "model_kwargs": model_kwargs, "window": window, "genome_build": "GRCh38",
                      "mode": args.mode, "seed": seed, "config": config,
                      "teacher_sha256": file_sha256(args.teacher), "data_provenance": provenance,
                      "split_fingerprint": fingerprint(splits), "cache_metadata": cache.metadata,
                      "selection": "minimum_validation_supervised_BCE", "n_epochs_run": len(history),
                      "clinical_use": "research_only", "calibration_status": "not_calibrated"}
        torch.save(checkpoint, output / "best_student.pt")
        if args.mode == "distill":
            torch.save(artifact, output / "distillation_training_state.pt")
        model = build_student(load_checkpoint(output / "best_student.pt"), device)
        save_history(output, history)
        write_json(output / "config.json", config)
    metrics = report_student(model, sequences, splits, output, device, provenance,
                             args.evaluate or output / "best_student.pt")
    print(json.dumps({"status": "evaluated" if args.evaluate else "trained", "mode": args.mode,
                      "test": metrics["test"], "performance_scope": provenance["performance_scope"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
