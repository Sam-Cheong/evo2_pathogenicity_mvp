"""Supervised and distilled student training after a completed teacher experiment.

Teacher information enters training only. Saved student checkpoints contain no
Evo 2 model or teacher projection, and inference imports no Evo 2 package.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

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


def model_state_sha256(model):
    """Fingerprint every initial parameter/buffer, including the random head."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Student state fingerprint requires tensor state entries")
        value = value.detach().cpu().contiguous()
        description = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(description).to_bytes(8, "big"))
        digest.update(description)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def validate_student_window(model, window, architecture):
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("Student DNA window must be a positive integer")
    maximum = getattr(model, "max_length", None)
    if architecture == "hyenadna_d256" and maximum is None:
        raise ValueError("Pretrained HyenaDNA must expose its verified DNA max_length")
    if maximum is not None and window > int(maximum):
        raise ValueError(f"Student window {window} exceeds the supported DNA length {maximum}; truncation is not allowed")


def initialize_student(student_config, window, seed):
    architecture = student_config.get("architecture", "cnn")
    if architecture == "hyenadna_d256":
        from .hyenadna import initialize_pretrained_student
        model, model_kwargs, initialization = initialize_pretrained_student(student_config, window)
        initialization = dict(initialization)
    elif architecture == "cnn":
        model_kwargs = {key: student_config.get(key, value)
                        for key, value in {"embedding_dim": 64, "channels": [32, 64, 96], "dropout": 0.2}.items()}
        model = TinyStudent(**model_kwargs)
        initialization = {"pretrained": False, "source": "seeded_random_initialization"}
    else:
        raise ValueError("Unsupported student architecture: " + str(architecture))
    validate_student_window(model, window, architecture)
    initialization["initial_model_state_sha256"] = model_state_sha256(model)
    protocol = {"architecture": architecture, "student_config": dict(student_config),
                "seed": int(seed), "window": int(window)}
    return model, model_kwargs, initialization, protocol


def student_training_settings(config):
    settings = training_options(config)
    hyena = config.get("student", {}).get("architecture", "cnn") == "hyenadna_d256"
    settings.setdefault("encoder_lr", 2e-5 if hyena else settings["lr"])
    settings.setdefault("head_lr", settings["lr"])
    settings.setdefault("eval_batch_size", 32)
    settings.setdefault("cpu_threads", 4)
    settings.setdefault("grad_clip_norm", 1.0)
    for key in ("encoder_lr", "head_lr", "grad_clip_norm"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    for key in ("eval_batch_size", "cpu_threads"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(key + " must be a positive integer")
    return settings


def build_student_optimizer(model, projector, settings, mode):
    if not hasattr(model, "encoder") or not hasattr(model, "head"):
        raise ValueError("Student must expose encoder and head modules for distinct learning rates")
    encoder = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder}
    head = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in encoder_ids]
    if not encoder or not head:
        raise ValueError("Student encoder and head must both have trainable parameters")
    groups = [{"params": encoder, "lr": settings["encoder_lr"], "name": "encoder"},
              {"params": head, "lr": settings["head_lr"], "name": "head_and_readout"}]
    if mode == "distill":
        groups.append({"params": list(projector.parameters()), "lr": settings["head_lr"], "name": "distillation_projector"})
    parameters = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("A parameter occurs in multiple student optimizer groups")
    optimizer = torch.optim.AdamW(groups, lr=settings["head_lr"], weight_decay=settings["weight_decay"])
    return optimizer, parameters


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


def teacher_targets(teacher, features, device, batch_size=32):
    targets = {}
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        for split, (values, _) in features.items():
            targets[split] = torch.cat([teacher(batch.to(device)).cpu() for batch in values.split(batch_size)])
    return targets


def fit_student(model, sequences, features, teacher, config, device, mode="distill"):
    if mode not in ("supervised", "distill"):
        raise ValueError("Unknown student training mode")
    options = student_training_settings(config)
    if torch.device(device).type == "cpu":
        torch.set_num_threads(options["cpu_threads"])
    architecture = config.get("student", {}).get("architecture", "cnn")
    validate_student_window(model, int(sequences["train"][0].shape[-1]), architecture)
    kd = {"alpha": 1.0, "beta": 0.1, "gamma": 0.5, "temperature": 2.0, "projection_dim": 128}
    kd.update(config.get("distillation", {}))
    if any(kd[name] < 0 for name in ("alpha", "beta", "gamma")) or kd["alpha"] <= 0:
        raise ValueError("Distillation weights must be nonnegative with alpha > 0")
    targets = teacher_targets(teacher, features, device, options["eval_batch_size"]) if mode == "distill" else None
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
    optimizer, parameters = build_student_optimizer(model, projector, options, mode)
    ref, alt, labels = sequences["train"]
    loader = DataLoader(TensorDataset(ref, alt, labels, torch.arange(len(labels))),
                        batch_size=options["batch_size"], shuffle=True)
    history, best_state, best_projector = [], None, None
    best_validation, bad_epochs, best_epoch = float("inf"), 0, None
    for epoch in range(1, options["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        projector.train()
        sums = {name: 0.0 for name in ("loss", "supervised", "feature", "prediction_kd")}
        gradient_norms = []
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
            gradient_norm = nn.utils.clip_grad_norm_(parameters, options["grad_clip_norm"], error_if_nonfinite=True)
            gradient_norms.append(float(gradient_norm))
            optimizer.step()
            for name, item in (("loss", loss), ("supervised", supervised), ("feature", feature), ("prediction_kd", prediction)):
                sums[name] += item.item() * len(batch_labels)
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(r.to(device), a.to(device))[0].cpu()
                                    for r, a in zip(sequences["val"][0].split(options["eval_batch_size"]),
                                                    sequences["val"][1].split(options["eval_batch_size"]))])
            validation_loss = F.binary_cross_entropy_with_logits(val_logits, sequences["val"][2]).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite student validation loss")
        row = {"epoch": epoch, "train_loss": sums["loss"] / len(labels), "val_loss": validation_loss,
               "train_supervised": sums["supervised"] / len(labels), "train_feature": sums["feature"] / len(labels),
               "train_prediction_kd": sums["prediction_kd"] / len(labels),
               "gradient_norm_before_clip": float(np.mean(gradient_norms)),
               "encoder_lr": options["encoder_lr"], "head_lr": options["head_lr"],
               "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        print(json.dumps({"event": "student_epoch", "mode": mode, **row}, allow_nan=False), flush=True)
        if validation_loss < best_validation - 1e-8:
            best_validation, bad_epochs = validation_loss, 0
            best_epoch = epoch
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
                         "best_epoch": best_epoch, "training_settings": options,
                         "projection_protocol": "fixed_seeded_random_teacher_projection_L2_normalized;train_only_feature_scaler"}
    return model.eval(), history, training_artifact


def report_student(model, sequences, splits, output, device, provenance, checkpoint_path, eval_batch_size=32):
    metrics, rows = {}, []
    with torch.inference_mode():
        for name in ("train", "val", "test"):
            ref, alt, labels = sequences[name]
            logits = torch.cat([model(r.to(device), a.to(device))[0].cpu()
                                for r, a in zip(ref.split(eval_batch_size), alt.split(eval_batch_size))])
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
    settings = student_training_settings(config)
    if device.type == "cpu":
        torch.set_num_threads(settings["cpu_threads"])
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
    features = feature_tensors(variants, splits, cache)
    teacher = build_teacher(teacher_checkpoint, device)
    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate)
        if checkpoint["split_fingerprint"] != fingerprint(splits) or checkpoint["data_provenance"] != provenance:
            raise ValueError("Evaluation dataset/split differs from student training")
        if checkpoint["teacher_sha256"] != file_sha256(args.teacher):
            raise ValueError("Student teacher lineage differs from the supplied experiment")
        if checkpoint["mode"] != args.mode:
            raise ValueError("Student evaluation mode differs from the saved training mode")
        model = build_student(checkpoint, device)
        architecture = checkpoint.get("architecture", "cnn")
        validate_student_window(model, window, architecture)
        sequences = sequence_tensors(variants, splits, args.fasta, window)
    else:
        student_config = config.get("student", {})
        architecture = student_config.get("architecture", "cnn")
        model, model_kwargs, initialization, initialization_protocol = initialize_student(student_config, window, seed)
        sequences = sequence_tensors(variants, splits, args.fasta, window)
        # Reset the stochastic training stream after architecture construction so
        # supervised and distilled runs start from the same shuffle/dropout seed.
        set_seed(seed)
        model, history, artifact = fit_student(model, sequences, features, teacher, config, device, args.mode)
        checkpoint = {"kind": "student", "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                      "architecture": architecture, "initialization": initialization,
                      "initialization_protocol": initialization_protocol,
                      "model_kwargs": model_kwargs, "window": window, "genome_build": "GRCh38",
                      "mode": args.mode, "seed": seed, "config": config,
                      "teacher_sha256": file_sha256(args.teacher), "data_provenance": provenance,
                      "split_fingerprint": fingerprint(splits), "cache_metadata": cache.metadata,
                      "selection": "minimum_validation_supervised_BCE", "n_epochs_run": len(history),
                      "best_epoch": artifact["best_epoch"], "training_settings": settings,
                      "clinical_use": "research_only", "calibration_status": "not_calibrated"}
        torch.save(checkpoint, output / "best_student.pt")
        if args.mode == "distill":
            torch.save(artifact, output / "distillation_training_state.pt")
        model = build_student(load_checkpoint(output / "best_student.pt"), device)
        save_history(output, history)
        write_json(output / "config.json", config)
    metrics = report_student(model, sequences, splits, output, device, provenance,
                             args.evaluate or output / "best_student.pt", settings["eval_batch_size"])
    print(json.dumps({"status": "evaluated" if args.evaluate else "trained", "mode": args.mode,
                      "test": metrics["test"], "performance_scope": provenance["performance_scope"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
