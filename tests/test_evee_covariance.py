import csv
import json

import numpy as np
import pytest
import torch

from evo2_mvp.evee_covariance import EVEEStyleCovarianceProbe, fit_evee_covariance


def test_covariance_uses_projected_second_moment_in_both_directions():
    probe = EVEEStyleCovarianceProbe(input_dim=2, projection_dim=1)
    with torch.no_grad():
        probe.left.weight.copy_(torch.tensor([[1.0, 0.0]]))
        probe.right.weight.copy_(torch.tensor([[0.0, 1.0]]))
        probe.readout.weight.fill_(1.0)
        probe.readout.bias.zero_()
    deltas = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[-2.0, 3.0], [2.0, -4.0]]]])
    torch.testing.assert_close(probe(deltas), torch.tensor([0.0]))
    deltas[0, 1, 1, 1] = 0.0
    torch.testing.assert_close(probe(deltas), torch.tensor([2.0]))


class TrackedDeltas:
    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.accessed.append(index)
        return self.values[index]


@pytest.fixture
def pilot_data():
    labels = np.tile([0, 1], 6)
    groups = [f"gene{i // 2}" for i in range(len(labels))]
    splits = {"train": list(range(6)), "val": [6, 7], "test": [8, 9, 10, 11]}
    deltas = np.random.default_rng(1).normal(size=(len(labels), 2, 3, 4)).astype(np.float32)
    return deltas, labels, groups, splits


def test_covariance_fit_keeps_holdout_separate_and_saves_replayable_model(pilot_data, tmp_path):
    deltas, labels, groups, splits = pilot_data
    means = np.repeat(labels[:, None, None], 8, axis=1).reshape(-1, 2, 4).astype(np.float32)
    means[splits["test"]] += 2.0
    variant_ids = [f"variant{i}" for i in range(len(labels))]
    tracked = TrackedDeltas(deltas)
    options = {"projection_dim": 2, "batch_size": 3, "epochs": 3, "selection": "validation_bce",
               "variant_ids": variant_ids, "means": means, "seed": 8}
    report = fit_evee_covariance(tracked, labels, groups, splits, output=tmp_path, **options)
    repeated = fit_evee_covariance(deltas, labels, groups, splits, **options)
    assert report == repeated
    assert tracked.accessed[-4:] == splits["test"]
    assert not set(tracked.accessed[:-4]) & set(splits["test"])
    assert report["selected_epoch"] == min(report["history"], key=lambda row: row["val_bce"])["epoch"]
    assert report["split_counts"]["test"] == {"variants": 4, "positives": 2, "gene_groups": 2}
    assert report["metrics"]["mean_pooled"]["test"]["auroc"] == 1.0
    assert report["config"]["mean_pooled_input"] == "full_downstream_mean"

    checkpoint = torch.load(tmp_path / "evee_covariance.pt", weights_only=True)
    torch.testing.assert_close(checkpoint["mean_pooled"]["mean"], torch.full((8,), 0.5, dtype=torch.float64))
    model = EVEEStyleCovarianceProbe(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        logits = model(torch.from_numpy(deltas)).numpy()
    with (tmp_path / "evee_covariance_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [row["variant_id"] for row in predictions] == variant_ids
    np.testing.assert_allclose([float(row["covariance_logit"]) for row in predictions], logits, rtol=1e-6)
    saved = json.loads((tmp_path / "evee_covariance_metrics.json").read_text(encoding="utf-8"))
    assert saved["split_fingerprint"] == checkpoint["split_fingerprint"]


def test_covariance_fit_rejects_cross_split_gene_group(pilot_data):
    deltas, labels, groups, splits = pilot_data
    groups[splits["test"][0]] = groups[0]
    with pytest.raises(ValueError, match="gene groups overlap"):
        fit_evee_covariance(deltas, labels, groups, splits)


def test_covariance_fit_rejects_nonfinite_delta(pilot_data):
    deltas, labels, groups, splits = pilot_data
    deltas[splits["train"][0], 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite ALT-minus-REF"):
        fit_evee_covariance(deltas, labels, groups, splits)
