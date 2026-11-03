import subprocess
import os
import sys
from pathlib import Path

import pytest
import torch

from evo2_mvp.inference import check_research_checkpoint
from evo2_mvp.models import TinyStudent, build_student, load_checkpoint, one_hot


def test_untrained_and_synthetic_inference_rejected():
    with pytest.raises(ValueError, match="trained real-data"):
        check_research_checkpoint({"data_provenance": {"synthetic": True}, "n_epochs_run": 20})
    with pytest.raises(ValueError, match="lineage"):
        check_research_checkpoint({"data_provenance": {"synthetic": False}, "n_epochs_run": 20})


def test_student_standalone_in_fresh_process(tmp_path):
    # Tests numerical execution and reload only, never pathogenicity performance.
    torch.manual_seed(2)
    model = TinyStudent().eval()
    ref, alt = one_hot(["ACGT" * 64]), one_hot(["ACGT" * 32 + "GCGT" + "ACGT" * 31])
    with torch.no_grad():
        expected = model(ref, alt)[0]
    path = tmp_path / "student_test_only.pt"
    torch.save({"kind": "student", "model_state": model.state_dict(), "model_kwargs": {}}, path)
    restored = build_student(load_checkpoint(path))
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(ref, alt)[0], rtol=0, atol=0)
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from evo2_mvp.models import load_checkpoint,build_student,one_hot; "
            "m=build_student(load_checkpoint(sys.argv[1])); "
            "v=m(one_hot(['ACGT'*64]),one_hot(['GCGT'+'ACGT'*63]))[0]; "
            "assert 'evo2' not in sys.modules; assert v.isfinite().all(); print('standalone_student_ok')")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=root, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert "standalone_student_ok" in result.stdout

