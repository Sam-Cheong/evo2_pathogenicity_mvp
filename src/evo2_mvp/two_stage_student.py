"""Label-free representation distillation followed by frozen-encoder probes.

The distill stage opens representation_train/val only. The probes stage opens
clinical train/val only. Clinical test is opened solely by the evaluate stage,
after all four predefined probes have completed. No Evo 2 client is imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from filelock import FileLock
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .metrics import binary_metrics, file_sha256, fingerprint, set_seed, write_csv
from .utils import save_json as write_json


ARCHITECTURE = "stripedhyena2_tiny"
PROBE_NAMES = ("random_linear", "distilled_linear", "random_mlp", "distilled_mlp")
REP_KEYS = {"ref", "alt", "teacher_ref", "teacher_alt", "variant_ids"}
PROBE_KEYS = {"ref", "alt", "labels", "variant_ids"}


def state_sha256(state):
    if isinstance(state, nn.Module):
        state = state.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Only tensor state can be fingerprinted")
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    return digest.hexdigest()


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_artifact(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_metadata(data_dir):
    path = Path(data_dir) / "metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("files_sha256"), dict) or not isinstance(value.get("cache_metadata"), dict):
        raise ValueError("Prepared data metadata lacks file hashes or teacher-cache provenance")
    return value, file_sha256(path)


def load_partition(data_dir, filename, metadata, representation):
    """Open exactly one allowed partition, verifying its own hash only."""
    allowed = {"representation_train.npz", "representation_val.npz"} if representation else {
        "probe_train.npz", "probe_val.npz", "probe_test.npz"}
    if filename not in allowed:
        raise ValueError("Partition is not permitted for this stage")
    path = Path(data_dir) / filename
    expected_hash = metadata["files_sha256"].get(filename)
    if not expected_hash or file_sha256(path) != expected_hash:
        raise ValueError("Prepared partition checksum mismatch: " + filename)
    expected_keys = REP_KEYS if representation else PROBE_KEYS
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("Representation inputs must contain no labels or extra fields" if representation else "Malformed probe partition")
        values = {key: np.array(archive[key], copy=True) for key in expected_keys}
    ids = values["variant_ids"]
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in {"U", "S"}:
        raise ValueError("Variant IDs must be nonempty, non-object string arrays")
    ids = [str(value.decode() if isinstance(value, bytes) else value) for value in ids.tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate variant IDs in partition")
    window = int(metadata["cache_metadata"].get("window", 256))
    for key in ("ref", "alt"):
        array = values[key]
        if array.shape != (len(ids), 4, window) or not np.isfinite(array).all():
            raise ValueError("DNA tensor does not match (N,4,window)")
        if not np.isin(array, [0, 1]).all() or not (array.sum(axis=1) == 1).all():
            raise ValueError("Expected strict A/C/G/T one-hot tensors")
        values[key] = torch.from_numpy(array.astype(np.float32, copy=False))
    # Exactly one substitution at the fixed reference coordinate is required.
    changed = (values["ref"] != values["alt"]).any(dim=1)
    if not (changed.sum(dim=1) == 1).all() or not changed[:, window // 2].all():
        raise ValueError("Prepared REF/ALT must differ once at the variant center")
    if representation:
        for key in ("teacher_ref", "teacher_alt"):
            array = values[key]
            if array.ndim != 2 or array.shape[0] != len(ids) or not array.shape[1] or not np.isfinite(array).all():
                raise ValueError("Teacher embeddings must be finite (N,D)")
            values[key] = torch.from_numpy(array.astype(np.float32, copy=False))
        if values["teacher_ref"].shape != values["teacher_alt"].shape:
            raise ValueError("Teacher REF and ALT dimensions differ")
        if metadata["cache_metadata"].get("feature_dim") != values["teacher_ref"].shape[1] * 3:
            raise ValueError("Teacher target dimension differs from the source feature cache")
    else:
        labels = values["labels"]
        if labels.shape != (len(ids),) or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("Probe labels must be binary and finite")
        values["labels"] = torch.from_numpy(labels.astype(np.float32, copy=False))
    values["variant_ids"] = ids
    return values


def split_ids(metadata, name):
    for candidate in (metadata.get("split_ids"), metadata.get("outer_split_ids"), metadata.get("splits")):
        if isinstance(candidate, dict) and name in candidate:
            return set(candidate[name])
    value = metadata.get("outer_" + name + "_ids")
    return set(value) if isinstance(value, list) else None


def validate_representation_ids(train, val, metadata):
    training, validation = set(train["variant_ids"]), set(val["variant_ids"])
    if training & validation:
        raise ValueError("Representation train/val IDs overlap")
    for name, values in (("train", training), ("val", validation)):
        listed = metadata.get("stage1_" + name + "_ids", metadata.get("representation_" + name + "_ids"))
        if listed is not None and set(listed) != values:
            raise ValueError("Representation partition differs from its prepared allowlist")
    outer_train = split_ids(metadata, "train")
    outer_val, outer_test = split_ids(metadata, "val"), split_ids(metadata, "test")
    if outer_train is None or outer_val is None or outer_test is None:
        raise ValueError("Label-free metadata must record outer train/val/test ID allowlists")
    if not (training | validation) <= outer_train or (training | validation) & (outer_val | outer_test):
        raise ValueError("Representation data includes a clinical validation/test ID")


def options(config, stage):
    defaults = {"distillation": {"epochs": 20, "batch_size": 16, "eval_batch_size": 32, "lr": 3e-4,
                                  "weight_decay": .01, "patience": 5, "grad_clip_norm": 1.,
                                  "std_floor": 1e-4, "ref_weight": .5, "alt_weight": .5, "delta_weight": 1.},
                "probe": {"epochs": 20, "batch_size": 32, "lr": 1e-3, "weight_decay": .01,
                          "patience": 5, "grad_clip_norm": 1., "std_floor": 1e-6,
                          "feature_batch_size": 32}}
    result = {**defaults[stage], **config.get(stage, {})}
    for key in ("epochs", "batch_size", "patience"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("eval_batch_size", "feature_batch_size"):
        if key in result and (not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 1):
            raise ValueError(key + " must be a positive integer")
    for key in ("lr", "std_floor", "grad_clip_norm"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    if stage == "distillation":
        for key in ("ref_weight", "alt_weight", "delta_weight"):
            if not np.isfinite(result[key]) or result[key] <= 0:
                raise ValueError("All representation loss weights must be finite and positive")
    return result


def model_kwargs(config, head_kind):
    values = {"d_model": 192, "ffn_dim": 384, "num_heads": 6, "window": 256, "dropout": .1}
    values.update(config.get("model", {}))
    values["head_kind"] = head_kind
    return values


def new_student(config, head_kind):
    from .stripedhyena2 import StripedHyena2Student
    kwargs = model_kwargs(config, head_kind)
    model = StripedHyena2Student(**kwargs)
    if sum(parameter.numel() for parameter in model.parameters()) >= 2_000_000:
        raise ValueError("Independent student encoder plus head must have fewer than 2M parameters")
    return model, kwargs


def weighted_representation_loss(components, settings):
    return sum(settings[key + "_weight"] * components[key] for key in ("ref", "alt", "delta"))


def teacher_statistics(train, std_floor):
    sequences = torch.cat([train["teacher_ref"], train["teacher_alt"]], dim=0)
    delta = train["teacher_alt"] - train["teacher_ref"]
    sequence_std = sequences.std(dim=0, unbiased=False)
    delta_std = delta.std(dim=0, unbiased=False)
    return {"sequence_mean": sequences.mean(dim=0), "sequence_scale": sequence_std.clamp_min(std_floor),
            "delta_mean": delta.mean(dim=0), "delta_scale": delta_std.clamp_min(std_floor),
            "sequence_clamped_dimensions": int((sequence_std < std_floor).sum()),
            "delta_clamped_dimensions": int((delta_std < std_floor).sum())}


def standardized_teacher(partition, statistics):
    return {"ref": (partition["teacher_ref"] - statistics["sequence_mean"]) / statistics["sequence_scale"],
            "alt": (partition["teacher_alt"] - statistics["sequence_mean"]) / statistics["sequence_scale"],
            "delta": (partition["teacher_alt"] - partition["teacher_ref"] - statistics["delta_mean"]) / statistics["delta_scale"]}


def constant_baseline(targets, settings):
    values = {key: float(value.square().mean()) for key, value in targets.items()}
    return {**values, "total": float(weighted_representation_loss(values, settings)),
            "protocol": "constant_training_mean_in_each_teacher_target_space"}


def representation_pass(model, projector, delta_projector, partition, targets, settings, device, optimizer=None, epoch_seed=0):
    training = optimizer is not None
    model.encoder.train(training)
    projector.train(training); delta_projector.train(training)
    dataset = TensorDataset(partition["ref"], partition["alt"], targets["ref"], targets["alt"], targets["delta"])
    generator = torch.Generator().manual_seed(epoch_seed)
    loader = DataLoader(dataset, batch_size=settings["batch_size"] if training else settings["eval_batch_size"],
                        shuffle=training, generator=generator if training else None)
    totals = {key: 0. for key in ("ref", "alt", "delta", "total")}
    embeddings, delta_embeddings, gradients = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for reference, alternate, target_ref, target_alt, target_delta in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            ref, alt = model.encode(reference.to(device)), model.encode(alternate.to(device))
            components = {"ref": F.mse_loss(projector(ref), target_ref.to(device)),
                          "alt": F.mse_loss(projector(alt), target_alt.to(device)),
                          "delta": F.mse_loss(delta_projector(alt - ref), target_delta.to(device))}
            loss = weighted_representation_loss(components, settings)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite representation loss")
            if training:
                loss.backward()
                parameters = list(model.encoder.parameters()) + list(projector.parameters()) + list(delta_projector.parameters())
                norm = nn.utils.clip_grad_norm_(parameters, settings["grad_clip_norm"], error_if_nonfinite=True)
                gradients.append(float(norm))
                optimizer.step()
            else:
                embeddings.extend([ref.detach().cpu(), alt.detach().cpu()])
                delta_embeddings.append((alt - ref).detach().cpu())
            n = len(reference)
            for key, value in {**components, "total": loss}.items():
                totals[key] += float(value.detach()) * n
    result = {key: value / len(dataset) for key, value in totals.items()}
    if embeddings:
        latent = torch.cat(embeddings, dim=0)
        result.update(embedding_mean_std=float(latent.std(dim=0, unbiased=False).mean()),
                      embedding_mean_variance=float(latent.var(dim=0, unbiased=False).mean()),
                      embedding_mean_norm=float(latent.norm(dim=-1).mean()))
        delta_latent = torch.cat(delta_embeddings, dim=0)
        result.update(delta_embedding_mean_std=float(delta_latent.std(dim=0, unbiased=False).mean()),
                      delta_embedding_mean_norm=float(delta_latent.norm(dim=-1).mean()))
    if gradients:
        result["gradient_norm_before_clip"] = float(np.mean(gradients))
    return result


def read_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def run_distillation(data_dir, output, config, device, resume=False):
    """No clinical probe file, ClinVar CSV, or trained teacher head is read here."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metadata, metadata_hash = load_metadata(data_dir)
    train = load_partition(data_dir, "representation_train.npz", metadata, True)
    val = load_partition(data_dir, "representation_val.npz", metadata, True)
    validate_representation_ids(train, val, metadata)
    settings = options(config, "distillation")
    seed = int(config.get("seed", 42)); set_seed(seed)
    protocol = {"configuration_sha256": fingerprint(config), "prepared_metadata_sha256": metadata_hash,
                "representation_files_sha256": {name: metadata["files_sha256"][name] for name in
                                                ("representation_train.npz", "representation_val.npz")},
                "train_ids_sha256": fingerprint(train["variant_ids"]), "val_ids_sha256": fingerprint(val["variant_ids"])}
    status_path = output / "distillation_status.json"
    if status_path.exists() and not resume:
        raise FileExistsError("This representation experiment already exists; use --resume or a new output directory")
    model, kwargs = new_student(config, "linear")
    if train["ref"].shape[-1] != kwargs["window"]:
        raise ValueError("Model window differs from prepared data")
    model.to(device)
    model.head.requires_grad_(False)
    dimension = train["teacher_ref"].shape[1]
    projector = nn.Linear(kwargs["d_model"], dimension).to(device)
    delta_projector = nn.Linear(kwargs["d_model"], dimension).to(device)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(projector.parameters()) + list(delta_projector.parameters()),
                                  lr=settings["lr"], weight_decay=settings["weight_decay"])
    statistics = teacher_statistics(train, settings["std_floor"])
    train_targets, val_targets = standardized_teacher(train, statistics), standardized_teacher(val, statistics)
    baselines = {"train": constant_baseline(train_targets, settings), "val": constant_baseline(val_targets, settings)}
    initial_path, best_path, last_path = (output / name for name in ("initial_encoder.pt", "best_encoder.pt", "last_distillation.pt"))
    history, best_loss, best_epoch, bad_epochs, start_epoch = [], float("inf"), None, 0, 1
    if resume and last_path.exists():
        state = load_artifact(last_path)
        if state["protocol"] != protocol:
            raise ValueError("Resume data/configuration differs from the original representation experiment")
        model.encoder.load_state_dict(state["encoder_state"])
        projector.load_state_dict(state["projector_state"]); delta_projector.load_state_dict(state["delta_projector_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history, best_loss, best_epoch, bad_epochs = state["history"], state["best_loss"], state["best_epoch"], state["bad_epochs"]
        start_epoch = state["epoch"] + 1
        initial = load_artifact(initial_path)
        if (initial.get("protocol") != protocol
                or state_sha256(initial["encoder_state"]) != initial.get("encoder_sha256")
                or state.get("initial_encoder_sha256") != initial.get("encoder_sha256")):
            raise ValueError("Resume initial encoder provenance or state is inconsistent")
    else:
        initial = {"artifact_kind": "initial_random_encoder", "architecture": ARCHITECTURE,
                   "encoder_state": cpu_state(model.encoder), "model_kwargs": kwargs, "seed": seed,
                   "protocol": protocol, "initialization": "random_not_pretrained", "clinical_use": "research_only"}
        initial["encoder_sha256"] = state_sha256(initial["encoder_state"])
        if initial_path.exists():
            previous = load_artifact(initial_path)
            if previous["protocol"] != protocol or previous["encoder_sha256"] != initial["encoder_sha256"]:
                raise ValueError("Existing initial encoder belongs to a different experiment")
        atomic_torch_save(initial, initial_path)
    training_parameters = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in projector.parameters()) + sum(p.numel() for p in delta_projector.parameters())
    status_base = {"stage": "representation_distillation", "protocol": protocol, "labels_used": False,
                   "clinical_validation_test_used": False, "n_train": len(train["variant_ids"]), "n_val": len(val["variant_ids"]),
                   "teacher_dim": dimension, "projector_count": 2,
                   "training_only_projection_parameters": sum(p.numel() for p in projector.parameters()) + sum(p.numel() for p in delta_projector.parameters()),
                   "total_training_parameters": training_parameters,
                   "inference_student_parameters": sum(p.numel() for p in model.parameters()), "constant_mean_baseline": baselines}
    write_json(status_path, {**status_base, "status": "running", "next_epoch": start_epoch})
    if bad_epochs < settings["patience"]:
        for epoch in range(start_epoch, settings["epochs"] + 1):
            began = time.perf_counter(); set_seed(seed * 100000 + epoch)
            training = representation_pass(model, projector, delta_projector, train, train_targets, settings, device, optimizer, seed + epoch)
            validation = representation_pass(model, projector, delta_projector, val, val_targets, settings, device)
            row = {"epoch": epoch, **{"train_" + k: v for k, v in training.items()},
                   **{"val_" + k: v for k, v in validation.items()}, "epoch_seconds": time.perf_counter() - began}
            history.append(row)
            print(json.dumps({"event": "representation_epoch", **row}, allow_nan=False), flush=True)
            if validation["total"] < best_loss - 1e-8:
                best_loss, best_epoch, bad_epochs = validation["total"], epoch, 0
                best = {"artifact_kind": "representation_distilled_encoder", "architecture": ARCHITECTURE,
                        "encoder_state": cpu_state(model.encoder), "model_kwargs": kwargs,
                        "initial_encoder_sha256": initial["encoder_sha256"], "protocol": protocol,
                        "selected_by": "representation_validation_loss_only", "best_epoch": epoch,
                        "best_validation": validation, "teacher_statistics": statistics,
                        "constant_mean_baseline": baselines, "projector_state": cpu_state(projector),
                        "delta_projector_state": cpu_state(delta_projector), "clinical_use": "research_only"}
                best["encoder_sha256"] = state_sha256(best["encoder_state"])
                atomic_torch_save(best, best_path)
            else:
                bad_epochs += 1
            atomic_torch_save({"artifact_kind": "representation_training_resume", "protocol": protocol,
                               "initial_encoder_sha256": initial["encoder_sha256"],
                               "encoder_state": cpu_state(model.encoder), "projector_state": cpu_state(projector),
                               "delta_projector_state": cpu_state(delta_projector), "optimizer_state": optimizer.state_dict(),
                               "epoch": epoch, "history": history, "best_loss": best_loss,
                               "best_epoch": best_epoch, "bad_epochs": bad_epochs}, last_path)
            write_csv(output / "distillation_curve.csv", history)
            write_json(status_path, {**status_base, "status": "running", "epoch": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss})
            if bad_epochs >= settings["patience"]:
                break
    if not best_path.is_file():
        raise RuntimeError("Representation training produced no valid best encoder")
    best = load_artifact(best_path)
    if best["protocol"] != protocol or state_sha256(best["encoder_state"]) != best["encoder_sha256"]:
        raise ValueError("Best representation encoder provenance or state is inconsistent")
    if history:
        write_csv(output / "distillation_curve.csv", history)
    write_json(status_path, {**status_base, "status": "completed", "epochs_run": len(history), "best_epoch": best["best_epoch"],
                            "best_validation": best["best_validation"], "initial_encoder_sha256": initial["encoder_sha256"],
                            "best_encoder_sha256": best["encoder_sha256"], "best_encoder_file_sha256": file_sha256(best_path)})
    return best


def validate_probe_ids(train, val, metadata):
    if set(train["variant_ids"]) & set(val["variant_ids"]):
        raise ValueError("Clinical probe train/validation IDs overlap")
    for name, partition in (("train", train), ("val", val)):
        expected = split_ids(metadata, name)
        if expected is None or set(partition["variant_ids"]) != expected:
            raise ValueError("Clinical probe IDs differ from the locked outer split: " + name)
    if len(torch.unique(train["labels"])) != 2:
        raise ValueError("Probe training requires both classes")


def freeze_encoder(model):
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    for parameter in model.encoder.parameters():
        parameter.grad = None


def verify_encoder_unchanged(model, expected):
    if model.encoder.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.encoder.parameters()):
        raise RuntimeError("Frozen encoder entered training or accumulated gradients")
    if state_sha256(model.encoder) != expected:
        raise RuntimeError("Frozen encoder parameters or buffers changed during probe training")


def frozen_features(model, partition, device, batch_size, expected_encoder_hash):
    verify_encoder_unchanged(model, expected_encoder_hash)
    values = []
    with torch.no_grad():
        for ref, alt in zip(partition["ref"].split(batch_size), partition["alt"].split(batch_size)):
            r, a = model.encode(ref.to(device)), model.encode(alt.to(device))
            value = torch.cat([r, a, a - r], dim=-1).detach().cpu()
            if value.ndim != 2 or not torch.isfinite(value).all():
                raise RuntimeError("Frozen student produced malformed/nonfinite features")
            values.append(value)
    verify_encoder_unchanged(model, expected_encoder_hash)
    return torch.cat(values, dim=0)


def cached_frozen_features(model, partition, source_name, source_hash, output, encoder_hash, device, batch_size):
    path = Path(output) / (source_name + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive["encoder_sha256"].item()) != encoder_hash or str(archive["source_sha256"].item()) != source_hash
                    or archive["variant_ids"].tolist() != partition["variant_ids"]):
                raise ValueError("Frozen feature cache belongs to different encoder/data")
            features = np.array(archive["features"], copy=True)
        if features.shape != (len(partition["variant_ids"]), model.feature_dim) or not np.isfinite(features).all():
            raise ValueError("Frozen feature cache is malformed/nonfinite")
        verify_encoder_unchanged(model, encoder_hash)
        return torch.from_numpy(features.astype(np.float32, copy=False))
    features = frozen_features(model, partition, device, batch_size, encoder_hash)
    atomic_npz_save(path, features=features.numpy(), variant_ids=np.asarray(partition["variant_ids"]),
                    encoder_sha256=np.array(encoder_hash), source_sha256=np.array(source_hash))
    return features


def fit_feature_scaler(model, training_features, floor):
    if training_features.ndim != 2 or not torch.isfinite(training_features).all():
        raise ValueError("Feature scaler requires finite training features")
    mean, scale = training_features.mean(0), training_features.std(0, unbiased=False).clamp_min(floor)
    with torch.no_grad():
        model.feature_mean.copy_(mean.to(model.feature_mean.device))
        model.feature_scale.copy_(scale.to(model.feature_scale.device))
    return {"fit_split": "probe_train_only", "mean_sha256": state_sha256({"mean": mean}),
            "scale_sha256": state_sha256({"scale": scale}), "std_floor": floor}


def probe_logits(model, features, device, batch_size):
    model.head.eval()
    chunks = []
    with torch.no_grad():
        for values in features.split(batch_size):
            values = values.to(device)
            logits = model.head((values - model.feature_mean) / model.feature_scale).squeeze(-1)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite frozen probe logits")
            chunks.append(logits.cpu())
    return torch.cat(chunks)


def checkpoint_base(config, metadata, metadata_hash, kwargs, arm, head_kind, encoder_hash, initial_hash, head_initial_hash, scaler):
    return {"kind": "student", "architecture": ARCHITECTURE, "model_kwargs": kwargs,
            "window": kwargs["window"], "genome_build": "GRCh38", "clinical_use": "research_only",
            "data_provenance": metadata["data_provenance"], "cache_metadata": metadata["cache_metadata"],
            "split_fingerprint": metadata["split_fingerprint"], "seed": int(config.get("seed", 42)),
            "config": config, "prepared_metadata_sha256": metadata_hash,
            "mode": "frozen_encoder_probe", "probe_arm": arm, "head_kind": head_kind,
            "encoder_sha256": encoder_hash, "common_initial_encoder_sha256": initial_hash,
            "head_initial_sha256": head_initial_hash, "feature_scaler": scaler,
            "encoder_initialization": "random_not_pretrained" if arm == "random" else "representation_distillation_only",
            "encoder_frozen": True, "calibration_status": "not_calibrated",
            "selection": "minimum_clinical_validation_BCE_only", "test_used_for_selection": False}


def fit_probe(model, training_features, validation_features, train, val, base, output, settings, device):
    """Only the head is trainable; the encoder remains in eval mode throughout."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    expected_encoder_hash = base["encoder_sha256"]
    verify_encoder_unchanged(model, expected_encoder_hash)
    model.head.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    history, best_loss, bad_epochs, best_epoch = [], float("inf"), 0, None
    for epoch in range(1, settings["epochs"] + 1):
        began = time.perf_counter(); set_seed(base["seed"] * 100000 + epoch)
        model.head.train()
        generator = torch.Generator().manual_seed(base["seed"] + epoch)
        loader = DataLoader(TensorDataset(training_features, train["labels"]), batch_size=settings["batch_size"], shuffle=True, generator=generator)
        total = 0.
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            values, labels = values.to(device), labels.to(device)
            logits = model.head((values - model.feature_mean) / model.feature_scale).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite frozen probe loss")
            loss.backward()
            if any(parameter.grad is not None for parameter in model.encoder.parameters()):
                raise RuntimeError("A frozen encoder received gradients during probe training")
            nn.utils.clip_grad_norm_(model.head.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        logits = probe_logits(model, validation_features, device, settings["feature_batch_size"])
        validation_loss = float(F.binary_cross_entropy_with_logits(logits, val["labels"]))
        row = {"epoch": epoch, "train_loss": total / len(train["labels"]), "val_loss": validation_loss,
               "epoch_seconds": time.perf_counter() - began}
        history.append(row)
        print(json.dumps({"event": "frozen_probe_epoch", "arm": base["probe_arm"], "head_kind": base["head_kind"], **row}, allow_nan=False), flush=True)
        verify_encoder_unchanged(model, expected_encoder_hash)
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, bad_epochs = validation_loss, epoch, 0
            checkpoint = {**base, "model_state": cpu_state(model), "n_epochs_run": epoch,
                          "best_epoch": epoch, "best_validation_loss": best_loss}
            atomic_torch_save(checkpoint, output / "best_student.pt")
        else:
            bad_epochs += 1
        write_csv(output / "training_curve.csv", history)
        write_json(output / "training_status.json", {"status": "running", "epoch": epoch, "best_epoch": best_epoch,
                                                    "encoder_unchanged": True, "clinical_test_opened": False})
        if bad_epochs >= settings["patience"]:
            break
    checkpoint = load_artifact(output / "best_student.pt")
    checkpoint["n_epochs_run"] = len(history)
    atomic_torch_save(checkpoint, output / "best_student.pt")
    model.load_state_dict(checkpoint["model_state"])
    verify_encoder_unchanged(model, expected_encoder_hash)
    metrics, rows = {}, []
    for split, features, partition in (("train", training_features, train), ("val", validation_features, val)):
        logits = probe_logits(model, features, device, settings["feature_batch_size"])
        scores = logits.sigmoid().numpy()
        metrics[split] = binary_metrics(partition["labels"].numpy(), scores)
        rows.extend({"variant_id": variant_id, "split": split, "label": int(label), "logit": float(logit), "score": float(score)}
                    for variant_id, label, logit, score in zip(partition["variant_ids"], partition["labels"], logits, scores))
    write_json(output / "validation_metrics.json", {**metrics, "clinical_test_opened": False, "provenance": base["data_provenance"]})
    write_csv(output / "train_validation_predictions.csv", rows)
    write_json(output / "training_status.json", {"status": "completed", "epochs_run": len(history), "best_epoch": best_epoch,
                                                "encoder_unchanged": True, "clinical_test_opened": False,
                                                "checkpoint_sha256": file_sha256(output / "best_student.pt")})
    return checkpoint


def load_encoder_pair(output, metadata_hash, config):
    output = Path(output)
    status = json.loads((output / "distillation_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError("A completed representation-distillation stage is required")
    initial, distilled = load_artifact(output / "initial_encoder.pt"), load_artifact(output / "best_encoder.pt")
    if (file_sha256(output / "best_encoder.pt") != status.get("best_encoder_file_sha256")
            or distilled.get("encoder_sha256") != status.get("best_encoder_sha256")
            or initial.get("encoder_sha256") != status.get("initial_encoder_sha256")):
        raise ValueError("Encoder artifacts differ from the completed representation-stage record")
    for state in (initial, distilled):
        if (state["protocol"]["prepared_metadata_sha256"] != metadata_hash
                or state["protocol"]["configuration_sha256"] != fingerprint(config)
                or state_sha256(state["encoder_state"]) != state["encoder_sha256"]):
            raise ValueError("Encoder artifact/data/configuration provenance mismatch")
    if initial["encoder_sha256"] != distilled["initial_encoder_sha256"]:
        raise ValueError("Distilled encoder does not descend from the baseline initialization")
    return initial, distilled


def run_probes(data_dir, output, config, device, resume=False, allow_synthetic=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metadata, metadata_hash = load_metadata(data_dir)
    if metadata.get("data_provenance", {}).get("synthetic") and not allow_synthetic:
        raise ValueError("Synthetic probe labels require the explicit smoke-test flag")
    initial, distilled = load_encoder_pair(output, metadata_hash, config)
    train = load_partition(data_dir, "probe_train.npz", metadata, False)
    val = load_partition(data_dir, "probe_val.npz", metadata, False)
    validate_probe_ids(train, val, metadata)
    status_path = output / "probes_status.json"
    if status_path.exists() and not resume:
        raise FileExistsError("Probe experiment already exists; use --resume or a new output directory")
    settings = options(config, "probe")
    status = {"stage": "frozen_probes", "status": "running", "prepared_metadata_sha256": metadata_hash,
              "configuration_sha256": fingerprint(config), "predefined_probes": list(PROBE_NAMES),
              "completed": [], "clinical_test_opened": False, "common_initial_encoder_sha256": initial["encoder_sha256"]}
    write_json(status_path, status)
    features_by_arm, encoder_by_arm = {}, {"random": initial, "distilled": distilled}
    for arm, artifact in encoder_by_arm.items():
        model, _ = new_student(config, "linear")
        model.encoder.load_state_dict(artifact["encoder_state"])
        model.to(device); freeze_encoder(model)
        features_by_arm[arm] = {}
        for split, partition in (("train", train), ("val", val)):
            features_by_arm[arm][split] = cached_frozen_features(model, partition, arm + "_" + split,
                metadata["files_sha256"]["probe_" + split + ".npz"], output / "frozen_features", artifact["encoder_sha256"],
                device, settings["feature_batch_size"])
    for index, head_kind in enumerate(("linear", "mlp")):
        # Construct once, then copy the same random head to both encoder arms.
        set_seed(int(config.get("seed", 42)) + 1000 + index)
        template, _ = new_student(config, head_kind)
        head_state, head_initial_hash = cpu_state(template.head), state_sha256(template.head)
        for arm in ("random", "distilled"):
            name = arm + "_" + head_kind
            directory = output / "probes" / name
            checkpoint_path = directory / "best_student.pt"
            artifact = encoder_by_arm[arm]
            model, kwargs = new_student(config, head_kind)
            model.encoder.load_state_dict(artifact["encoder_state"]); model.head.load_state_dict(head_state)
            model.to(device); freeze_encoder(model)
            scaler = fit_feature_scaler(model, features_by_arm[arm]["train"], settings["std_floor"])
            scaler["fit_variant_ids_sha256"] = fingerprint(train["variant_ids"])
            base = checkpoint_base(config, metadata, metadata_hash, kwargs, arm, head_kind, artifact["encoder_sha256"],
                                   initial["encoder_sha256"], head_initial_hash, scaler)
            completed_path = directory / "training_status.json"
            if resume and checkpoint_path.exists() and completed_path.exists() and json.loads(completed_path.read_text())["status"] == "completed":
                existing = load_artifact(checkpoint_path)
                for key, value in base.items():
                    if existing.get(key) != value:
                        raise ValueError("Existing probe differs from the controlled experiment: " + name + "/" + key)
                if state_sha256({k.removeprefix("encoder."): v for k, v in existing["model_state"].items() if k.startswith("encoder.")}) != artifact["encoder_sha256"]:
                    raise ValueError("Existing probe encoder fingerprint differs")
            else:
                fit_probe(model, features_by_arm[arm]["train"], features_by_arm[arm]["val"], train, val, base, directory, settings, device)
            status["completed"].append(name)
            write_json(status_path, status)
    status.update(status="completed", all_probes_trained_before_test=True,
                  checkpoints_sha256={name: file_sha256(output / "probes" / name / "best_student.pt") for name in PROBE_NAMES})
    write_json(status_path, status)
    return status


def run_evaluation(data_dir, output, config, device):
    """This is the only code path allowed to open probe_test.npz."""
    output = Path(output)
    metadata, metadata_hash = load_metadata(data_dir)
    status = json.loads((output / "probes_status.json").read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or set(status.get("completed", [])) != set(PROBE_NAMES)
            or status.get("all_probes_trained_before_test") is not True
            or status.get("prepared_metadata_sha256") != metadata_hash
            or status.get("configuration_sha256") != fingerprint(config)):
        raise ValueError("All four predefined probes must complete on the same experiment before clinical test is opened")
    checkpoints = {}
    for name in PROBE_NAMES:
        path = output / "probes" / name / "best_student.pt"
        if file_sha256(path) != status["checkpoints_sha256"][name]:
            raise ValueError("A probe checkpoint changed after training completion")
        checkpoint = load_artifact(path)
        if (checkpoint.get("kind") != "student" or checkpoint["prepared_metadata_sha256"] != metadata_hash
                or checkpoint["split_fingerprint"] != metadata["split_fingerprint"] or checkpoint["config"] != config):
            raise ValueError("Probe evaluation provenance mismatch")
        checkpoints[name] = checkpoint
    for head_kind in ("linear", "mlp"):
        left, right = (checkpoints[arm + "_" + head_kind] for arm in ("random", "distilled"))
        for key in ("head_initial_sha256", "common_initial_encoder_sha256", "model_kwargs"):
            if left[key] != right[key]:
                raise ValueError("Frozen-probe control mismatch: " + key)
    # All provenance/training gates precede this single clinical test read.
    test = load_partition(data_dir, "probe_test.npz", metadata, False)
    expected = split_ids(metadata, "test")
    if expected is None or set(test["variant_ids"]) != expected:
        raise ValueError("Clinical test IDs differ from the locked outer test split")
    settings, reports, all_rows, features_by_arm = options(config, "probe"), {}, [], {}
    for name in PROBE_NAMES:
        checkpoint = checkpoints[name]
        arm = checkpoint["probe_arm"]
        model, kwargs = new_student(config, checkpoint["head_kind"])
        if kwargs != checkpoint["model_kwargs"]:
            raise ValueError("Evaluation model construction differs from the saved probe")
        model.load_state_dict(checkpoint["model_state"]); model.to(device); freeze_encoder(model)
        if arm not in features_by_arm:
            features_by_arm[arm] = cached_frozen_features(model, test, arm + "_test", metadata["files_sha256"]["probe_test.npz"],
                output / "frozen_features", checkpoint["encoder_sha256"], device, settings["feature_batch_size"])
        logits = probe_logits(model, features_by_arm[arm], device, settings["feature_batch_size"])
        scores = logits.sigmoid().numpy()
        metrics = binary_metrics(test["labels"].numpy(), scores)
        reports[name] = {"test": metrics, "encoder_frozen": True, "head_kind": checkpoint["head_kind"],
                         "probe_arm": arm, "checkpoint_sha256": status["checkpoints_sha256"][name]}
        rows = [{"variant_id": variant_id, "split": "test", "label": int(label), "logit": float(logit), "score": float(score)}
                for variant_id, label, logit, score in zip(test["variant_ids"], test["labels"], logits, scores)]
        directory = output / "probes" / name
        write_json(directory / "test_metrics.json", metrics)
        write_json(directory / "test_provenance.json", {"checkpoint_sha256": status["checkpoints_sha256"][name],
                                                        "provenance": metadata["data_provenance"], "test_used_for_selection": False})
        write_csv(directory / "test_predictions.csv", rows)
        all_rows.extend({"probe": name, **row} for row in rows)
        verify_encoder_unchanged(model, checkpoint["encoder_sha256"])
    report = {"status": "evaluated_all_predefined_probes", "models": reports, "same_independent_test_set": True,
              "test_ids_sha256": fingerprint(test["variant_ids"]), "prepared_metadata_sha256": metadata_hash,
              "representation_training_ids_only": True, "test_used_for_selection": False,
              "selection": "no_selection_between_the_four_models_using_test", "clinical_use": "research_only"}
    write_json(output / "two_stage_test_comparison.json", report)
    write_csv(output / "two_stage_test_predictions.csv", all_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("distill", "probes", "evaluate"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/stripedhyena2_two_stage.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-synthetic-smoke-test", action="store_true")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    threads = config.get("cpu_threads", 4)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if device.type == "cpu":
        torch.set_num_threads(threads)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".experiment.lock"), timeout=0):
        try:
            if args.stage == "distill":
                result = run_distillation(args.data_dir, output, config, device, args.resume)
                print(json.dumps({"status": "completed", "stage": "distill", "best_epoch": result["best_epoch"],
                                  "encoder_sha256": result["encoder_sha256"]}, allow_nan=False))
            elif args.stage == "probes":
                result = run_probes(args.data_dir, output, config, device, args.resume, args.allow_synthetic_smoke_test)
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                run_evaluation(args.data_dir, output, config, device)
        except Exception as exc:
            write_json(output / (args.stage + "_failure.json"), {"status": "failed", "stage": args.stage,
                "error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
            raise


if __name__ == "__main__":
    main()
