"""Restart guarantees use actual child processes, never fabricated training."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def support():
    path = Path(__file__).resolve().parents[1] / "scripts/hyenadna_notebook_support.py"
    spec = importlib.util.spec_from_file_location("notebook_support_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_restarts_without_rerunning_child_and_rejects_changed_artifact(tmp_path, support):
    script = tmp_path / "child with space.py"
    script.write_text("from pathlib import Path\np=Path('artifact.txt')\np.write_text(p.read_text()+'x' if p.exists() else 'x')\n", encoding="utf-8")
    identity = {"identity": "fixed-test-inputs"}
    first = support.run_stage(tmp_path, tmp_path / "run", "child", [script.name], identity, ["artifact.txt"])
    second = support.run_stage(tmp_path, tmp_path / "run", "child", [script.name], identity, ["artifact.txt"])
    assert first == second and (tmp_path / "artifact.txt").read_text() == "x"
    (tmp_path / "artifact.txt").write_text("modified")
    with pytest.raises(RuntimeError, match="modified"):
        support.run_stage(tmp_path, tmp_path / "run", "child", [script.name], identity, ["artifact.txt"])


def test_failed_process_never_creates_completed_stage_or_hides_error(tmp_path, support):
    script = tmp_path / "fail.py"
    script.write_text("print('deliberate failure')\nraise RuntimeError('failure')\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as error:
        support.run_stage(tmp_path, tmp_path / "run", "fail", [script.name], {"identity": "x"}, [])
    assert error.value.returncode != 0
    assert not (tmp_path / "run/fail.stage.json").exists()
    progress = json.loads((tmp_path / "run/fail.progress.json").read_text())
    assert progress["status"] == "failed"
    assert "deliberate failure" in (tmp_path / "run/fail.log").read_text()


def test_zero_exit_with_missing_output_is_failure(tmp_path, support):
    with pytest.raises(RuntimeError, match="artifact missing"):
        support.run_stage(tmp_path, tmp_path / "run", "missing", ["-c", "pass"], {"identity": "x"}, ["absent.txt"])
    assert not (tmp_path / "run/missing.stage.json").exists()


def test_changed_input_identity_prevents_reuse(tmp_path, support):
    support.run_stage(tmp_path, tmp_path / "run", "done", ["-c", "pass"], {"identity": "x"}, [])
    with pytest.raises(RuntimeError, match="different inputs"):
        support.run_stage(tmp_path, tmp_path / "run", "done", ["-c", "pass"], {"identity": "y"}, [])
