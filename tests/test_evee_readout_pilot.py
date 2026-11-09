from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn.functional import one_hot
from threadpoolctl import threadpool_limits

from evo2_mvp import evee_readout_pilot as pilot
from evo2_mvp.stripedhyena2 import StripedHyena2Encoder
from evo2_mvp.two_stage_student import state_sha256


def test_projected_gram_is_symmetric_uncentered_and_variant_center_is_first():
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    values = pilot.readouts(tokens, torch.eye(2))
    torch.testing.assert_close(values["center"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(values["mean"], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(values["gram"], torch.tensor([[5.0, 7.0, 10.0]]))
    for zero in pilot.readouts(torch.zeros_like(tokens), torch.eye(2)).values():
        assert not zero.any()


def test_specificity_threshold_and_paired_bootstrap():
    labels = np.array([0] * 20 + [1] * 20)
    good = np.arange(40) / 40
    poor = good[::-1]
    threshold = pilot.choose_threshold(labels, good)
    assert pilot.metrics_at_threshold(labels, good, threshold)["specificity"] >= 0.9
    part = {"labels": torch.from_numpy(labels), "groups": np.arange(40)}
    individual, difference, _ = pilot.cluster_intervals(
        part, {"good": (good, threshold), "poor": (poor, threshold)}, {"primary": ("good", "poor")}, 42, 100)
    assert individual["good"]["auprc"] == pytest.approx([1.0, 1.0])
    assert difference["primary"]["delta_auprc"][0] > 0


def synthetic_partition(seed, size):
    ref = torch.randint(4, (size, 256), generator=torch.Generator().manual_seed(seed))
    alt = ref.clone()
    alt[:, 128] = (alt[:, 128] + 1) % 4
    return {
        "ref": one_hot(ref, 4).permute(0, 2, 1).float(),
        "alt": one_hot(alt, 4).permute(0, 2, 1).float(),
        "variant_ids": [f"{seed}-{i}" for i in range(size)],
        "labels": torch.arange(size) % 2,
        "groups": np.array([f"{seed}-{i}" for i in range(size)]),
    }


@pytest.mark.parametrize("gate_passes", [False, True])
def test_pilot_locks_ranks_before_test_and_reloads_offline(tmp_path, monkeypatch, gate_passes):
    torch.set_num_threads(2)
    kwargs = {"d_model": 32, "ffn_dim": 48, "num_heads": 4, "window": 256, "dropout": .1}
    artifacts = {}
    for seed, arm in enumerate(pilot.ARMS):
        torch.manual_seed(seed)
        encoder = StripedHyena2Encoder(**kwargs)
        artifacts[arm] = {
            "model_kwargs": {**kwargs, "head_kind": "linear"},
            "encoder_state": encoder.state_dict(),
            "encoder_sha256": state_sha256(encoder),
        }
    train, val, test = (synthetic_partition(seed, 30) for seed in (1, 2, 3))
    provenance = {key: "test_fixture" for key in (
        "prepared_metadata_sha256", "variants_sha256", "split_fingerprint")}
    inputs = SimpleNamespace(train=train, val=val, encoder_artifacts=artifacts, provenance=provenance)
    monkeypatch.setattr(pilot, "load_pilot_inputs", lambda *args: inputs)
    test_reads = []

    def open_test(*args):
        test_reads.append(True)
        return test

    monkeypatch.setattr(pilot, "load_historical_test", open_test)
    with pytest.raises(ValueError, match="lock"):
        pilot.describe_test(None, None, tmp_path, "cpu")
    assert not test_reads

    # Set the primary interval explicitly to cover both branches. Keep all
    # feature extraction, grouped CV, checkpoint serialization and inference real.
    actual_intervals = pilot.cluster_intervals
    actual_fit_rank = pilot.fit_rank

    def controlled_intervals(partition, predictions, pairs, seed, draws):
        results, deltas, diagnostics = actual_intervals(partition, predictions, pairs, seed, draws)
        if partition is val and "distilled_r16" in deltas:
            deltas["distilled_r16"]["delta_auprc"] = [0.01, 0.20] if gate_passes else [-0.01, 0.20]
        return results, deltas, diagnostics

    def controlled_rank(rank, *args, **kwargs):
        results, predictions, checkpoints = actual_fit_rank(rank, *args, **kwargs)
        if rank == 16:
            results["distilled_gram_r16"]["validation"]["auprc"] = 0.8
            results["distilled_mean_r16"]["validation"]["auprc"] = 0.7
        return results, predictions, checkpoints

    monkeypatch.setattr(pilot, "cluster_intervals", controlled_intervals)
    monkeypatch.setattr(pilot, "fit_rank", controlled_rank)
    with threadpool_limits(limits=2):
        report = pilot.run_fit(None, None, None, {}, tmp_path, "cpu", 42, 50, 16)
    assert not test_reads
    assert report["rank32_triggered"] is gate_passes
    assert len(report["models"]) == (12 if gate_passes else 6)
    pca_hashes = {r["shared_pca_sha256"] for r in report["models"].values() if r["rank"] == 16}
    assert len(pca_hashes) == 1

    path = tmp_path / "models" / "distilled_gram_r16.pt"
    model_a, checkpoint = pilot.load_pilot_model(path)
    model_b, _ = pilot.load_pilot_model(path)
    assert state_sha256(model_a.encoder) == checkpoint["encoder_sha256"] == artifacts["distilled"]["encoder_sha256"]
    with torch.inference_mode():
        sequence = (test["ref"][:2], test["alt"][:2])
        torch.testing.assert_close(model_a(*sequence), model_b(*sequence), atol=0, rtol=0)
    assert not any(p.requires_grad for p in model_a.parameters())

    described = pilot.describe_test(None, None, tmp_path, "cpu")
    assert test_reads == [True]
    assert described["test_used_for_selection"] is False
    assert described["models"]["distilled_gram_r16"]["test"]["classification_threshold"] == checkpoint["threshold"]
    assert len(described["models"]) == len(report["models"])

    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed after lock"):
        pilot.describe_test(None, None, tmp_path, "cpu")
    assert test_reads == [True]

