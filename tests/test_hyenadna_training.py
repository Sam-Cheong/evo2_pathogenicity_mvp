"""Weight-free integration tests for the pretrained-student training protocol.

Fixture encoders below are explicitly synthetic and never represent HyenaDNA
checkpoint verification or pathogenicity performance.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evo2_mvp import distill_student as training
from evo2_mvp.models import TeacherMLP, one_hot


class FixtureStudent(nn.Module):
    """Small test-only encoder with the same forward/optimizer interface."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv1d(4, 4, 1)
        self.head = nn.Linear(12, 1)
        self.feature_dim = 12
        self.max_length = 16
        self.calls = []

    def forward(self, ref, alt):
        self.calls.append((self.training, ref.shape[0]))
        r, a = self.encoder(ref).mean(-1), self.encoder(alt).mean(-1)
        fused = torch.cat([r, a, a - r], dim=-1)
        return self.head(fused).squeeze(-1), fused


def test_complete_initial_state_fingerprint_includes_random_head_and_buffers():
    torch.manual_seed(42)
    first = FixtureStudent()
    first.register_buffer("test_buffer", torch.tensor(1))
    second = copy.deepcopy(first)
    assert training.model_state_sha256(first) == training.model_state_sha256(second)
    with torch.no_grad():
        second.head.bias.add_(1)
    assert training.model_state_sha256(first) != training.model_state_sha256(second)
    second = copy.deepcopy(first)
    second.test_buffer.add_(1)
    assert training.model_state_sha256(first) != training.model_state_sha256(second)


def test_pretrained_initializer_contract_and_window_gate(monkeypatch):
    calls = []
    def fixture_initializer(config, window):
        calls.append((config, window))
        return FixtureStudent(), {"fixture": True}, {"source": "unit_test_fixture_only", "revision": "fixture"}
    monkeypatch.setitem(sys.modules, "evo2_mvp.hyenadna", SimpleNamespace(initialize_pretrained_student=fixture_initializer))
    config = {"architecture": "hyenadna_d256", "pretrained_dir": "unit_test_no_weights", "pooling": "variant_center"}
    model, kwargs, metadata, protocol = training.initialize_student(config, 16, 42)
    assert calls == [(config, 16)]
    assert kwargs == {"fixture": True}
    assert metadata["initial_model_state_sha256"] == training.model_state_sha256(model)
    assert protocol == {"architecture": "hyenadna_d256", "student_config": config, "window": 16, "seed": 42}
    with pytest.raises(ValueError, match="exceeds"):
        training.initialize_student(config, 32, 42)
    with pytest.raises(ValueError, match="Unsupported"):
        training.initialize_student({"architecture": "unverified_replacement"}, 16, 42)


def test_distinct_learning_rates_and_nonduplicated_parameters():
    settings = training.student_training_settings({"student": {"architecture": "hyenadna_d256"}})
    assert settings["encoder_lr"] == 2e-5
    assert settings["head_lr"] == 1e-3
    assert settings["eval_batch_size"] == 32
    assert settings["cpu_threads"] == 4
    assert settings["grad_clip_norm"] == 1
    model, projector = FixtureStudent(), nn.Linear(12, 8)
    optimizer, parameters = training.build_student_optimizer(model, projector, settings, "distill")
    assert [(g["name"], g["lr"]) for g in optimizer.param_groups] == [
        ("encoder", 2e-5), ("head_and_readout", 1e-3), ("distillation_projector", 1e-3)]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in list(model.parameters()) + list(projector.parameters())}
    for field in ("encoder_lr", "head_lr", "eval_batch_size", "cpu_threads", "grad_clip_norm"):
        with pytest.raises(ValueError):
            training.student_training_settings({"training": {field: 0}})


@pytest.mark.parametrize("mode", ["supervised", "distill"])
def test_fixture_training_clips_gradients_batches_validation_and_logs_epochs(mode, monkeypatch, capsys):
    previous_threads = torch.get_num_threads()
    try:
        torch.manual_seed(42)
        model = FixtureStudent()
        ref, alt = one_hot(["ACGT" * 4] * 20), one_hot(["ATGT" + "ACGT" * 3] * 20)
        labels = torch.tensor([0., 1.] * 10)
        raw = torch.arange(120, dtype=torch.float32).reshape(20, 6) / 120
        slices = {"train": slice(0, 8), "val": slice(8, 15), "test": slice(15, 20)}
        sequences = {name: (ref[sl], alt[sl], labels[sl]) for name, sl in slices.items()}
        features = {name: (raw[sl], labels[sl]) for name, sl in slices.items()}
        teacher = TeacherMLP(6, hidden_dims=(8, 4), dropout=0)
        teacher.fit_normalizer(features["train"][0])
        clip_calls = []
        original_clip = nn.utils.clip_grad_norm_
        def observed_clip(parameters, maximum, **kwargs):
            clip_calls.append(maximum)
            return original_clip(parameters, maximum, **kwargs)
        monkeypatch.setattr(nn.utils, "clip_grad_norm_", observed_clip)
        config = {"seed": 42, "student": {"architecture": "hyenadna_d256"},
                  "training": {"epochs": 2, "batch_size": 4, "eval_batch_size": 3,
                               "patience": 2, "cpu_threads": 1, "grad_clip_norm": 1.0}}
        model, history, artifact = training.fit_student(model, sequences, features, teacher, config, "cpu", mode)
        assert len(history) == 2 and clip_calls == [1., 1., 1., 1.]
        assert max(size for is_train, size in model.calls if not is_train) == 3
        assert artifact["best_epoch"] in (1, 2)
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        assert all(row["gradient_norm_before_clip"] >= 0 and row["epoch_seconds"] >= 0 for row in history)
        assert all(not p.requires_grad for p in teacher.parameters()) if mode == "distill" else True
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [e["event"] for e in events] == ["student_epoch", "student_epoch"]
        assert all(e["mode"] == mode for e in events)
    finally:
        torch.set_num_threads(previous_threads)


def comparison_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("hyenadna_comparison_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_checkpoint():
    return {"architecture": "hyenadna_d256", "model_kwargs": {"fixture": True}, "seed": 42, "window": 256,
            "config": {"student": {"architecture": "hyenadna_d256"}, "training": {"encoder_lr": 2e-5}},
            "initialization": {"revision": "fixture_revision", "config_sha256": "b" * 64,
                               "initial_model_state_sha256": "a" * 64},
            "initialization_protocol": {"architecture": "hyenadna_d256", "seed": 42, "window": 256},
            "training_settings": {"encoder_lr": 2e-5, "head_lr": 1e-3}}


def test_comparison_requires_same_pretrained_revision_full_state_and_configuration():
    module = comparison_module()
    left = comparison_checkpoint()
    assert module.validate_student_comparison(left, copy.deepcopy(left))["matched"] is True
    for mutation in ("architecture", "state", "revision", "config", "protocol", "optimizer"):
        right = copy.deepcopy(left)
        if mutation == "architecture":
            right["architecture"] = "cnn"
        elif mutation == "state":
            right["initialization"]["initial_model_state_sha256"] = "c" * 64
        elif mutation == "revision":
            right["initialization"]["revision"] = "different_revision"
        elif mutation == "config":
            right["config"]["training"]["encoder_lr"] = 1e-3
        elif mutation == "protocol":
            right["initialization_protocol"]["window"] = 512
        else:
            right["training_settings"]["head_lr"] = 2e-3
        with pytest.raises(ValueError):
            module.validate_student_comparison(left, right)


def test_legacy_cnn_comparison_remains_available_but_not_initialization_verified():
    module = comparison_module()
    legacy = {"model_kwargs": {}, "seed": 42, "window": 256, "config": {"student": {}}}
    result = module.validate_student_comparison(legacy, copy.deepcopy(legacy))
    assert result["architecture"] == "cnn" and result["matched"] is False
    assert result["status"] == "unverified_legacy_cnn"
    with pytest.raises(ValueError, match="initialization provenance"):
        incomplete = comparison_checkpoint()
        incomplete.pop("initialization")
        module.validate_student_comparison(incomplete, copy.deepcopy(incomplete))
