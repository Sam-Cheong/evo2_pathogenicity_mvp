"""Fit and evaluate an EVEE-style probe over precomputed ALT-minus-REF activations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .metrics import binary_metrics, fingerprint, set_seed, write_csv
from .train_teacher_head import choose_device
from .two_stage_student import atomic_torch_save, cpu_state
from .utils import save_json


class EVEEStyleCovarianceProbe(nn.Module):
    """Project the two directions' tokens, pool their second moments, and classify."""

    def __init__(self, input_dim: int = 4096, projection_dim: int = 64):
        super().__init__()
        self.left = nn.Linear(input_dim, projection_dim, bias=False)
        self.right = nn.Linear(input_dim, projection_dim, bias=False)
        self.readout = nn.Linear(projection_dim * projection_dim, 1)

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        tokens = deltas.flatten(1, 2)
        left = self.left(tokens)
        right = self.right(tokens)
        pooled = left.transpose(1, 2).bmm(right) / tokens.shape[1]
        return self.readout(pooled.flatten(1)).squeeze(-1)


def _validate_partitions(deltas, labels, groups, splits, variant_ids):
    n = len(deltas)
    labels = np.asarray(labels)
    groups = list(groups)
    if labels.shape != (n,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("A binary label is required for every delta tensor")
    if len(groups) != n or any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("A nonempty connected gene group is required for every delta tensor")
    variant_ids = [str(i) for i in range(n)] if variant_ids is None else list(variant_ids)
    if (len(variant_ids) != n or any(not isinstance(key, str) or not key.strip() for key in variant_ids)
            or len(set(variant_ids)) != n):
        raise ValueError("A unique variant ID is required for every delta tensor")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Provide explicit train, val, and test index partitions")

    indices = {}
    seen = set()
    for split in ("train", "val", "test"):
        partition = np.asarray(splits[split])
        if (partition.ndim != 1 or not len(partition) or partition.dtype.kind not in "iu"
                or (partition < 0).any() or (partition >= n).any()):
            raise ValueError(f"Invalid sample indices in the {split} partition")
        indices[split] = partition
        current = {groups[i] for i in partition}
        if current & seen:
            raise ValueError(f"Connected gene groups overlap the {split} partition")
        seen.update(current)
    if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(n)):
        raise ValueError("Every variant must appear in exactly one partition")
    if np.unique(labels[indices["train"]]).size != 2:
        raise ValueError("Covariance and mean-pooled probes require both classes in training")
    return labels.astype(np.int64), groups, indices, variant_ids


def _sample(values, i, shape=None):
    value = values[int(i)]
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise ValueError(f"Expected floating-point features at index {i}")
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
    else:
        value = np.asarray(value)
        if value.dtype.kind != "f":
            raise ValueError(f"Expected floating-point features at index {i}")
        tensor = torch.tensor(value, dtype=torch.float32)
    if shape is not None and tensor.shape != shape:
        raise ValueError(f"Inconsistent feature shape at index {i}. Expected {tuple(shape)}, got {tuple(tensor.shape)}")
    return tensor


def _batches(deltas, indices, sample_shape, batch_size, device):
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        batch = torch.stack([_sample(deltas, i, sample_shape) for i in rows]).to(device)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError(f"Nonfinite ALT-minus-REF activations in sample indices {rows.tolist()}")
        yield rows, batch


@torch.no_grad()
def _evaluate(probe, batches, means=None, *, collect_means=False):
    probe.eval()
    logits, pooled = [], []
    for rows, batch in batches:
        logits.append(probe(batch).cpu())
        if collect_means:
            if means is None:
                mean_batch = batch.mean(dim=2).cpu()
            else:
                shape = (batch.shape[1], batch.shape[-1])
                mean_batch = torch.stack([_sample(means, i, shape) for i in rows])
            if not bool(torch.isfinite(mean_batch).all()):
                raise ValueError(f"Nonfinite mean-pooled activations in sample indices {rows.tolist()}")
            pooled.append(mean_batch.flatten(1).numpy())
    logits = torch.cat(logits)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Covariance probe produced nonfinite logits")
    return logits, np.concatenate(pooled) if collect_means else None


def fit_evee_covariance(
    deltas, labels, groups, splits, *, variant_ids=None, means=None, output=None,
    device="cpu", batch_size=4, projection_dim=64, epochs=1, learning_rate=0.005,
    weight_decay=0.1, seed=42, selection="last",
):
    """Train using ClinVar labels, then compare the selected model and mean baseline.

    ``deltas[i]`` returns ``[2, K, D]`` with directions first. Optional ``means[i]``
    returns the downstream mean for each direction as ``[2, D]``. Both are
    indexed in the same order as labels, groups and variant IDs. Only a batch
    of token activations is kept in memory. The fitted mean baseline uses
    standardized concatenated direction means with fixed L2 regularization.
    """
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size, projection_dim)):
        raise ValueError("Epochs, batch size and projection dimension must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Learning rate must be finite and positive and weight decay finite and nonnegative")
    if selection not in ("last", "validation_bce"):
        raise ValueError("Checkpoint selection must be 'last' or 'validation_bce'")
    labels, groups, splits, variant_ids = _validate_partitions(deltas, labels, groups, splits, variant_ids)
    if means is not None and len(means) != len(deltas):
        raise ValueError("Optional downstream means must align with all delta tensors")
    sample_shape = _sample(deltas, splits["train"][0]).shape
    if len(sample_shape) != 3 or sample_shape[0] != 2 or min(sample_shape) < 1:
        raise ValueError("ALT-minus-REF activations must have shape [2 directions, K positions, D channels]")

    device = choose_device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Select CPU or CUDA for the covariance probe")
    set_seed(seed)
    random = np.random.default_rng(seed)
    model_kwargs = {"input_dim": sample_shape[-1], "projection_dim": projection_dim}
    probe = EVEEStyleCovarianceProbe(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    selected_state, selected_epoch, best_loss = None, None, float("inf")
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        train_order = random.permutation(splits["train"])
        for rows, batch in _batches(deltas, train_order, sample_shape, batch_size, device):
            target = torch.as_tensor(labels[rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(probe(batch), target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite covariance probe training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rows)

        val_indices = splits["val"]
        val_logits, _ = _evaluate(probe, _batches(deltas, val_indices, sample_shape, batch_size, device))
        val_target = torch.from_numpy(labels[val_indices]).double()
        val_loss = F.binary_cross_entropy_with_logits(val_logits.double(), val_target).item()
        history.append({"epoch": epoch, "train_bce": total_loss / len(train_order), "val_bce": val_loss})
        if (selection == "last" and epoch == epochs) or (selection == "validation_bce" and val_loss < best_loss):
            selected_state, selected_epoch, best_loss = cpu_state(probe), epoch, val_loss

    probe.load_state_dict(selected_state)
    metrics = {"covariance": {}, "mean_pooled": {}}
    predictions = []
    for name, indices in splits.items():
        logits, features = _evaluate(
            probe, _batches(deltas, indices, sample_shape, batch_size, device), means, collect_means=True,
        )
        targets = labels[indices]
        if name == "train":
            scaler = StandardScaler().fit(features)
            baseline = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(features), targets)
            if baseline.n_iter_[0] >= baseline.max_iter:
                raise RuntimeError("Mean-pooled logistic regression did not converge")
        scaled = scaler.transform(features)
        mean_logits = baseline.decision_function(scaled)
        mean_scores = baseline.predict_proba(scaled)[:, 1]
        cov_scores = logits.double().sigmoid().numpy()
        metrics["covariance"][name] = binary_metrics(targets, cov_scores)
        metrics["mean_pooled"][name] = binary_metrics(targets, mean_scores)
        predictions.extend(
            {"variant_id": variant_ids[i], "split": name, "gene_group": groups[i], "label": int(labels[i]),
             "covariance_logit": cov_logit, "covariance_score": float(cov_score),
             "mean_pooled_logit": float(mean_logit), "mean_pooled_score": float(mean_score)}
            for i, cov_logit, cov_score, mean_logit, mean_score in zip(
                indices, logits.tolist(), cov_scores, mean_logits, mean_scores,
            )
        )

    config = {
        "device": str(device), "batch_size": batch_size, "epochs": epochs, "seed": seed,
        "input_shape": list(sample_shape), "projection_dim": projection_dim,
        "optimizer": "AdamW", "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
        "covariance_pooling": "uncentered_cross_moment_shared_projections_over_both_directions",
        "mean_pooled_input": "full_downstream_mean" if means is not None else "mean_of_retained_positions",
        "mean_pooled_l2_C": 1.0, "checkpoint_selection": selection,
    }
    report = {
        "status": "complete", "method": "evee_style_covariance_pilot", "config": config,
        "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
        "selected_epoch": selected_epoch, "history": history, "metrics": metrics,
        "split_counts": {
            name: {"variants": len(indices), "positives": int(labels[indices].sum()),
                   "gene_groups": len({groups[i] for i in indices})}
            for name, indices in splits.items()
        },
        "split_fingerprint": fingerprint({name: [variant_ids[i] for i in indices] for name, indices in splits.items()}),
    }
    if output is not None:
        output = Path(output)
        atomic_torch_save({
            "artifact_kind": "evee_style_covariance_probe", "schema_version": 1,
            "config": config, "model_kwargs": model_kwargs, "model_state": selected_state,
            "selected_epoch": selected_epoch, "split_fingerprint": report["split_fingerprint"],
            "mean_pooled": {"mean": torch.from_numpy(scaler.mean_), "scale": torch.from_numpy(scaler.scale_),
                            "weight": torch.from_numpy(baseline.coef_[0]), "bias": torch.from_numpy(baseline.intercept_)},
        }, output / "evee_covariance.pt")
        write_csv(output / "evee_covariance_predictions.csv", predictions)
        save_json(output / "evee_covariance_metrics.json", report)
    return {**report, "predictions": predictions}
