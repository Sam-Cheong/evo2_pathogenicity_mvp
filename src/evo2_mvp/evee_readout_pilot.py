"""Frozen Student token readouts inspired by EVEE, with an explicit historical-test lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from .evee_pilot_data import grouped_folds, load_historical_test, load_pilot_inputs
from .metrics import binary_metrics, file_sha256, fingerprint, write_csv
from .stripedhyena2 import StripedHyena2Encoder
from .two_stage_student import atomic_torch_save, load_artifact, state_sha256
from .utils import load_config, save_json


ARMS = ("random", "distilled")
READOUTS = ("center", "mean", "gram")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
TOKEN_COUNT = 32
PRIMARY_RANK = 16
SECONDARY_RANK = 32
TARGET_SPECIFICITY = 0.90
PROTOCOL_VERSION = 1
LOGISTIC_SOLVER = "newton-cholesky"


def model_name(arm, readout, rank):
    return f"{arm}_{readout}_r{rank}"


def freeze_encoder(encoder, expected_sha):
    encoder.eval().requires_grad_(False)
    verify_frozen(encoder, expected_sha)
    return encoder


def verify_frozen(encoder, expected_sha):
    if encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise RuntimeError("Pilot encoder must remain frozen and in evaluation mode")
    if state_sha256(encoder) != expected_sha:
        raise RuntimeError("Pilot encoder state differs from the original checkpoint")


def encode_deltas(encoder, ref, alt):
    center = ref.shape[-1] // 2
    if ref.shape != alt.shape or center + TOKEN_COUNT > ref.shape[-1]:
        raise ValueError("Pilot requires aligned REF/ALT and 32 tokens starting at the variant")
    positions = slice(center, center + TOKEN_COUNT)
    return encoder(alt)[:, positions] - encoder(ref)[:, positions]


@torch.inference_mode()
def extract_deltas(encoder, partition, device, batch_size, expected_sha):
    verify_frozen(encoder, expected_sha)
    chunks = [
        encode_deltas(encoder, ref.to(device), alt.to(device)).cpu()
        for ref, alt in zip(partition["ref"].split(batch_size), partition["alt"].split(batch_size))
    ]
    tokens = torch.cat(chunks)
    if not torch.isfinite(tokens).all():
        raise RuntimeError("Student produced nonfinite token differences")
    verify_frozen(encoder, expected_sha)
    return tokens


def encoder_from_artifact(artifact, device):
    kwargs = {key: value for key, value in artifact["model_kwargs"].items() if key != "head_kind"}
    encoder = StripedHyena2Encoder(**kwargs)
    encoder.load_state_dict(artifact["encoder_state"], strict=True)
    return freeze_encoder(encoder.to(device), artifact["encoder_sha256"]), kwargs


def fit_basis(tokens_by_arm, rank):
    """Fit common PCA directions on the two equally weighted training arms."""
    total = torch.cat([tokens_by_arm[arm].flatten(0, 1) for arm in ARMS]).double()
    if rank > min(total.shape[0] - 1, total.shape[1]):
        raise ValueError("Pilot PCA rank exceeds the available training samples or hidden dimension")
    total -= total.mean(dim=0)
    _, vectors = torch.linalg.eigh(total.T @ total / (len(total) - 1))
    basis = vectors[:, -rank:].flip(1).T.contiguous()
    # A deterministic sign convention preserves coordinate identity in saved artifacts.
    anchors = basis.gather(1, basis.abs().argmax(dim=1, keepdim=True))
    return (basis * anchors.sign()).to(torch.float32)


def readouts(tokens, basis):
    # Fit PCA axes on centered training data, then project without subtracting
    # a global mean. This keeps the zero point of the REF/ALT delta and its Gram.
    z = tokens @ basis.T
    return {kind: projected_readout(z, kind) for kind in READOUTS}


def projected_readout(z, kind):
    if kind == "center":
        return z[:, 0]
    if kind == "mean":
        return z.mean(dim=1)
    if kind != "gram":
        raise ValueError("Unknown pilot readout")
    gram = z.transpose(1, 2) @ z / z.shape[1]
    upper = torch.triu_indices(z.shape[-1], z.shape[-1], device=z.device)
    return gram[:, upper[0], upper[1]]


def fit_logistic_classifier(features, labels, c):
    classifier = LogisticRegression(C=c, max_iter=256, solver=LOGISTIC_SOLVER, tol=1e-7)
    classifier.fit(features, labels)
    if classifier.n_iter_[0] >= classifier.max_iter:
        raise RuntimeError(f"Pilot logistic regression did not converge (C={c:g})")
    return classifier


def logistic_fit(features, labels, c):
    features = np.asarray(features, dtype=np.float64)
    scaler = StandardScaler().fit(features)
    classifier = fit_logistic_classifier(scaler.transform(features), labels, c)
    return scaler, classifier


def cv_regularization(tokens, train, folds, rank):
    labels = train["labels"].numpy()
    scores = {(arm, kind): np.zeros((len(C_GRID), len(folds))) for arm in ARMS for kind in READOUTS}
    for fold_index, (fit, heldout) in enumerate(folds):
        basis = fit_basis({arm: tokens[arm][fit] for arm in ARMS}, rank)
        for arm in ARMS:
            x_train, x_heldout = readouts(tokens[arm][fit], basis), readouts(tokens[arm][heldout], basis)
            for kind in READOUTS:
                fit_features = np.asarray(x_train[kind], dtype=np.float64)
                heldout_features = np.asarray(x_heldout[kind], dtype=np.float64)
                scaler = StandardScaler().fit(fit_features)
                a = scaler.transform(fit_features)
                b = scaler.transform(heldout_features)
                for i, c in enumerate(C_GRID):
                    try:
                        estimator = fit_logistic_classifier(a, labels[fit], c)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Grouped CV fold {fold_index + 1}, {arm}/{kind}: {exc}") from exc
                    scores[arm, kind][i, fold_index] = average_precision_score(
                        labels[heldout], estimator.predict_proba(b)[:, 1])
    result = {}
    for key, values in scores.items():
        mean = values.mean(axis=1)
        result[key] = {
            "selected_C": C_GRID[int(np.argmax(mean))],
            "mean_ap_by_C": dict(zip(map(str, C_GRID), mean.tolist())),
            "fold_ap_by_C": dict(zip(map(str, C_GRID), values.tolist())),
        }
    return result


class PilotStudent(nn.Module):
    """Self-contained frozen encoder, PCA transform, scaler and logistic readout."""

    def __init__(self, encoder, checkpoint):
        super().__init__()
        self.encoder = encoder
        self.readout = checkpoint["readout"]
        for name in ("basis", "mean", "scale", "weight", "bias"):
            self.register_buffer(name, checkpoint[name])
        self.eval().requires_grad_(False)

    def token_logits(self, tokens):
        features = projected_readout(tokens @ self.basis.T, self.readout).double()
        return ((features - self.mean) / self.scale) @ self.weight + self.bias

    def forward(self, ref, alt):
        return self.token_logits(encode_deltas(self.encoder, ref, alt))


def load_pilot_model(path, device="cpu"):
    checkpoint = load_artifact(path)
    if checkpoint.get("artifact_kind") != "evee_inspired_frozen_student_readout":
        raise ValueError("Expected an EVEE-inspired pilot checkpoint")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported pilot checkpoint version")
    encoder, _ = encoder_from_artifact(checkpoint, device)
    return PilotStudent(encoder, checkpoint).to(device).eval(), checkpoint


def choose_threshold(labels, scores):
    """Maximize recall on validation subject to the preregistered specificity."""
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= (1 - TARGET_SPECIFICITY) + 1e-12)
    chosen = eligible[int(np.argmax(tpr[eligible]))]
    threshold = thresholds[chosen]
    if not np.isfinite(threshold):
        threshold = np.nextafter(np.max(scores), np.inf)
    return float(threshold)


def metrics_at_threshold(labels, scores, threshold):
    metrics = binary_metrics(labels, scores)
    tn, fp, fn, tp = np.bincount(2 * labels + (scores >= threshold), minlength=4).tolist()
    metrics.update({
        "accuracy": (tn + tp) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "classification_threshold": threshold,
        "threshold_use": "fixed_from_validation_for_research_metrics_only",
    })
    undefined = metrics["undefined_metrics"]
    undefined.pop("precision", None)
    if metrics["precision"] is None:
        undefined["precision"] = "No positive predictions are present"
    if metrics["specificity"] is None:
        undefined["specificity"] = "No negative true labels are present"
    return metrics


def bootstrap_metrics(labels, scores, threshold):
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    tp = labels[order].cumsum()[ends]
    fp = ends + 1 - tp
    ap = (np.diff(np.r_[0, tp]) / tp[-1] * tp / (tp + fp)).sum()
    auc = np.trapz(np.r_[0, tp] / tp[-1], np.r_[0, fp] / fp[-1])
    positive = labels == 1
    prediction = scores >= threshold
    return np.array([ap, auc, np.mean((scores - labels) ** 2),
                     prediction[positive].mean(), (~prediction[~positive]).mean()])


def cluster_intervals(partition, predictions, pairs, seed, draws):
    """Paired percentile intervals conditional on the fitted encoders and readouts."""
    labels = partition["labels"].numpy().astype(int)
    group_values = np.asarray(partition["groups"])
    groups = [np.flatnonzero(group_values == group) for group in np.unique(group_values)]
    positive_groups = sum(bool(labels[group].any()) for group in groups)
    if len(groups) < 2 or positive_groups < 2 or sum(bool((labels[g] == 0).any()) for g in groups) < 2:
        raise ValueError("Group bootstrap requires at least two groups carrying each class")
    random = np.random.default_rng(seed)
    replicates = {name: [] for name in predictions}
    for _ in range(draws):
        sample = np.concatenate([groups[i] for i in random.integers(0, len(groups), len(groups))])
        y = labels[sample]
        if y.min() == y.max():
            continue
        for name, (scores, threshold) in predictions.items():
            replicates[name].append(bootstrap_metrics(y, scores[sample], threshold))
    valid = len(next(iter(replicates.values())))
    if valid < 0.95 * draws:
        raise ValueError("Too many single-class gene-group bootstrap draws")

    def interval(values):
        return np.quantile(values, (0.025, 0.975)).tolist()

    per_model = {
        name: {metric: interval(column) for metric, column in zip(
            ("auprc", "auroc", "brier_score", "recall", "specificity"), np.asarray(values).T)}
        for name, values in replicates.items()
    }
    differences = {
        key: {metric: interval(column) for metric, column in zip(
            ("delta_auprc", "delta_auroc"),
            (np.asarray(replicates[candidate]) - np.asarray(replicates[baseline]))[:, :2].T)}
        for key, (candidate, baseline) in pairs.items()
    }
    return per_model, differences, {
        "method": "paired_gene_connected_component_percentile_bootstrap",
        "conditional_on_fixed_fitted_models_and_validation_thresholds": True,
        "draws_requested": draws, "draws_used": valid, "seed": seed,
        "n_groups": len(groups), "positive_groups": positive_groups,
    }


def rank_pairs(ranks):
    return {f"{arm}_r{rank}": (model_name(arm, "gram", rank), model_name(arm, "mean", rank))
            for rank in ranks for arm in ARMS}


def prediction_rows(partition, predictions, split):
    return [
        {"model": name, "split": split, "variant_id": variant_id, "label": int(y),
         "score": float(score), "threshold": threshold, "predicted_label": int(score >= threshold)}
        for name, (scores, threshold) in predictions.items()
        for variant_id, y, score in zip(partition["variant_ids"], partition["labels"], scores)
    ]



def fit_rank(rank, tokens, inputs, folds, output, parameter_counts):
    train_tokens = {arm: tokens[arm]["train"] for arm in ARMS}
    search = cv_regularization(train_tokens, inputs.train, folds, rank)
    basis = fit_basis(train_tokens, rank)
    basis_sha = state_sha256({"basis": basis})
    train_y, val_y = (part["labels"].numpy().astype(int) for part in (inputs.train, inputs.val))
    results, predictions, checkpoints = {}, {}, {}
    for arm in ARMS:
        artifact = inputs.encoder_artifacts[arm]
        train = readouts(train_tokens[arm], basis)
        validation = readouts(tokens[arm]["val"], basis)
        encoder_parameters = parameter_counts[arm]
        encoder_stored = sum(value.numel() for value in artifact["encoder_state"].values())
        for kind in READOUTS:
            name = model_name(arm, kind, rank)
            c = search[arm, kind]["selected_C"]
            scaler, estimator = logistic_fit(train[kind], train_y, c)
            checkpoint = {
                "artifact_kind": "evee_inspired_frozen_student_readout",
                "protocol_version": PROTOCOL_VERSION,
                "architecture": "stripedhyena2_tiny",
                "arm": arm, "rank": rank, "readout": kind, "clinical_use": "research_only",
                "model_kwargs": artifact["model_kwargs"], "encoder_state": artifact["encoder_state"],
                "encoder_sha256": artifact["encoder_sha256"], "basis": basis, "pca_sha256": basis_sha,
                "mean": torch.from_numpy(scaler.mean_), "scale": torch.from_numpy(scaler.scale_),
                "weight": torch.from_numpy(estimator.coef_[0]), "bias": torch.from_numpy(estimator.intercept_[0:1]),
                "data_provenance": inputs.provenance,
            }
            # Use the same double-precision affine map as deployment for threshold fitting.
            x = validation[kind].double()
            logits = ((x - checkpoint["mean"]) / checkpoint["scale"]) @ checkpoint["weight"] + checkpoint["bias"]
            scores = logits.sigmoid().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("Pilot validation scores are nonfinite")
            threshold = choose_threshold(val_y, scores)
            checkpoint["threshold"] = threshold
            path = output / "models" / f"{name}.pt"
            atomic_torch_save(checkpoint, path)
            checkpoints[name] = file_sha256(path)
            stored = encoder_stored + sum(checkpoint[key].numel() for key in ("basis", "mean", "scale", "weight", "bias"))
            if stored >= 2_000_000:
                raise ValueError("Pilot inference footprint must remain below 2M stored scalar values")
            results[name] = {
                "arm": arm, "rank": rank, "readout": kind, "feature_width": train[kind].shape[1],
                "encoder_parameters": encoder_parameters,
                "logistic_parameters": train[kind].shape[1] + 1,
                "inference_scalar_values_including_pca_and_scaler": stored,
                "encoder_sha256_before_and_after": artifact["encoder_sha256"], "encoder_unchanged": True,
                "shared_pca_sha256": basis_sha, "checkpoint_sha256": checkpoints[name],
                "regularization": search[arm, kind], "validation": metrics_at_threshold(val_y, scores, threshold),
            }
            predictions[name] = (scores, threshold)
    return results, predictions, checkpoints


def run_fit(data_dir, encoders_dir, variants, encoder_config, output, device, seed, draws, batch_size):
    output = Path(output)
    if (output / "pilot_lock.json").exists():
        raise FileExistsError("This pilot is already locked. Use a new output directory for another protocol.")
    inputs = load_pilot_inputs(data_dir, encoders_dir, variants, encoder_config)
    folds = grouped_folds(inputs.train, n_splits=5, seed=seed)
    tokens, parameter_counts = {}, {}
    for arm in ARMS:
        artifact = inputs.encoder_artifacts[arm]
        encoder, kwargs = encoder_from_artifact(artifact, device)
        if kwargs["window"] != 256:
            raise ValueError("Pilot protocol requires the original 256-bp context")
        parameter_counts[arm] = sum(p.numel() for p in encoder.parameters())
        tokens[arm] = {
            split: extract_deltas(encoder, part, device, batch_size, artifact["encoder_sha256"])
            for split, part in (("train", inputs.train), ("val", inputs.val))
        }
        verify_frozen(encoder, artifact["encoder_sha256"])

    results, predictions, checkpoints = fit_rank(PRIMARY_RANK, tokens, inputs, folds, output, parameter_counts)
    pairs = rank_pairs((PRIMARY_RANK,))
    per_model, differences, bootstrap = cluster_intervals(inputs.val, predictions, pairs, seed, draws)
    key = f"distilled_r{PRIMARY_RANK}"
    candidate, baseline = pairs[key]
    delta_ap = results[candidate]["validation"]["auprc"] - results[baseline]["validation"]["auprc"]
    delta_ci = differences[key]["delta_auprc"]
    passed = delta_ap > 0 and delta_ci[0] > 0
    primary = {
        "candidate": candidate, "baseline": baseline, "split": "validation",
        "delta_auprc": delta_ap, "delta_auprc_95_ci": delta_ci,
        "rule": "delta_auprc_positive_and_paired_group_bootstrap_95_ci_lower_bound_positive",
        "passed": passed,
    }
    ranks = [PRIMARY_RANK]
    if passed:
        ranks.append(SECONDARY_RANK)
        secondary, secondary_predictions, secondary_checkpoints = fit_rank(SECONDARY_RANK, tokens, inputs, folds, output, parameter_counts)
        results.update(secondary)
        predictions.update(secondary_predictions)
        checkpoints.update(secondary_checkpoints)
        intervals, paired, _ = cluster_intervals(
            inputs.val, secondary_predictions, rank_pairs((SECONDARY_RANK,)), seed, draws)
        per_model.update(intervals)
        differences.update(paired)

    for name, result in results.items():
        result["validation_group_bootstrap_95_ci"] = per_model[name]
    protocol = {
        "version": PROTOCOL_VERSION, "seed": seed, "orientation": "forward_reference",
        "window": 256, "token_start_zero_based": 128, "tokens": TOKEN_COUNT,
        "pca": "shared_across_arms_and_readouts_centered_fit_untranslated_projection",
        "gram": "upper_triangle_of_uncentered_symmetric_ZtZ_divided_by_token_count",
        "arms": list(ARMS), "readouts": list(READOUTS), "ranks": ranks,
        "inner_cv": "stratified_gene_connected_group_5fold_ap_for_logistic_L2",
        "cv_heldout_id_hashes": [fingerprint(sorted(inputs.train["variant_ids"][i] for i in val_idx))
                                 for _, val_idx in folds],
        "C_grid": list(C_GRID), "logistic_solver": LOGISTIC_SOLVER,
        "target_validation_specificity": TARGET_SPECIFICITY,
        "threshold_rule": "maximize_validation_recall_at_minimum_specificity_ties_choose_higher_threshold",
        "checkpoint_selection_uses_historical_test": False,
    }
    report = {
        "status": "validation_complete", "protocol": protocol, "primary": primary,
        "rank32_triggered": passed, "models": results, "paired_gram_minus_mean_95_ci": differences,
        "bootstrap": bootstrap, "provenance": inputs.provenance,
        "historical_test_opened": False, "new_independent_test_cohort_available": False,
        "clinical_use": "research_only",
    }
    validation_report = output / "pilot_validation.json"
    save_json(validation_report, report)
    write_csv(output / "validation_predictions.csv", prediction_rows(inputs.val, predictions, "val"))
    lock = {
        "status": "locked_for_historical_test", "protocol": protocol, "primary": primary,
        "checkpoint_hashes": checkpoints, "validation_report_sha256": file_sha256(validation_report),
        "prepared_metadata_sha256": inputs.provenance["prepared_metadata_sha256"],
        "variants_sha256": inputs.provenance["variants_sha256"],
        "split_fingerprint": inputs.provenance["split_fingerprint"],
        "batch_size": batch_size, "bootstrap_draws": draws,
    }
    save_json(output / "pilot_lock.json", {**lock, "lock_fingerprint": fingerprint(lock)})
    return report


def read_locked_candidates(output):
    lock_path = output / "pilot_lock.json"
    if not lock_path.is_file():
        raise ValueError("Complete and lock the validation pilot before opening historical test")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared_fingerprint = lock.pop("lock_fingerprint", None)
    if lock.get("status") != "locked_for_historical_test" or fingerprint(lock) != declared_fingerprint:
        raise ValueError("Pilot lock is incomplete or was modified")
    ranks = [PRIMARY_RANK, SECONDARY_RANK] if lock["primary"]["passed"] else [PRIMARY_RANK]
    expected = {model_name(arm, readout, rank) for rank in ranks for arm in ARMS for readout in READOUTS}
    if lock["protocol"]["version"] != PROTOCOL_VERSION or lock["protocol"]["ranks"] != ranks:
        raise ValueError("Pilot lock has unexpected protocol or rank decision")
    if set(lock["checkpoint_hashes"]) != expected:
        raise ValueError("Pilot lock has missing or extra candidates")
    if file_sha256(output / "pilot_validation.json") != lock["validation_report_sha256"]:
        raise ValueError("Pilot validation report differs from the locked report")
    paths = {name: output / "models" / f"{name}.pt" for name in lock["checkpoint_hashes"]}
    for name, path in paths.items():
        if file_sha256(path) != lock["checkpoint_hashes"][name]:
            raise ValueError(f"Pilot checkpoint changed after lock: {name}")
    return {**lock, "lock_fingerprint": declared_fingerprint}, paths


def describe_test(data_dir, variants, output, device):
    output = Path(output)
    lock, paths = read_locked_candidates(output)
    # Read only after the validation report, decision and every weight file are checked.
    partition = load_historical_test(data_dir, variants, lock)
    labels = partition["labels"].numpy().astype(int)
    tokens, predictions, results = {}, {}, {}
    for name, path in paths.items():
        model, checkpoint = load_pilot_model(path, device)
        arm, sha = checkpoint["arm"], checkpoint["encoder_sha256"]
        key = (arm, sha)
        if key not in tokens:
            tokens[key] = extract_deltas(model.encoder, partition, device, lock["batch_size"], sha).to(device)
        with torch.inference_mode():
            scores = model.token_logits(tokens[key]).sigmoid().cpu().numpy()
        verify_frozen(model.encoder, sha)
        threshold = checkpoint["threshold"]
        results[name] = {"test": metrics_at_threshold(labels, scores, threshold), "encoder_sha256": sha}
        predictions[name] = (scores, threshold)
    per_model, differences, bootstrap = cluster_intervals(
        partition, predictions, rank_pairs(lock["protocol"]["ranks"]),
        lock["protocol"]["seed"], lock["bootstrap_draws"])
    for name, result in results.items():
        result["historical_test_group_bootstrap_95_ci"] = per_model[name]
    report = {
        "status": "historical_test_described", "scope": "previously_examined_test_set_descriptive_only",
        "new_independent_test_cohort_available": False, "test_used_for_selection": False,
        "validation_lock_fingerprint": lock["lock_fingerprint"], "primary_from_validation": lock["primary"],
        "models": results, "paired_gram_minus_mean_95_ci": differences, "bootstrap": bootstrap,
        "test_ids_sha256": fingerprint(sorted(partition["variant_ids"])), "clinical_use": "research_only",
    }
    write_csv(output / "historical_test_predictions.csv", prediction_rows(partition, predictions, "historical_test"))
    save_json(output / "historical_test_description.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "describe-test"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoders", type=Path)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, default=Path("configs/stripedhyena2_two_stage.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1 or args.bootstrap_draws < 1000 or args.seed < 0:
        parser.error("Require positive threads/batch size, at least 1000 bootstrap draws and a nonnegative seed")
    if args.stage == "fit" and args.encoders is None:
        parser.error("--encoders is required with --stage fit")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".pilot.writer.lock"), timeout=0), threadpool_limits(limits=args.threads):
        if args.stage == "fit":
            report = run_fit(args.data_dir, args.encoders, args.variants, load_config(args.encoder_config),
                             args.output, device, args.seed, args.bootstrap_draws, args.batch_size)
        else:
            report = describe_test(args.data_dir, args.variants, args.output, device)
    print(json.dumps({key: value for key, value in report.items() if key in {"status", "primary", "rank32_triggered"}},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

