"""Host-side runners: command construction and guards (no Docker, no GPU)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[1] / "run"
sys.path.insert(0, str(RUN))
import predict  # noqa: E402
import run_phase  # noqa: E402
from containers import experiments_of_phase  # noqa: E402


def test_phase_membership():
    assert "exp07_real100" in experiments_of_phase(1)
    assert experiments_of_phase(1, only=["exp07_real100"]) == ["exp07_real100"]
    assert all(e.startswith("px_") for e in experiments_of_phase(4))


def test_extra_seed_runs_use_the_data_of_the_first_seed():
    assert predict.data_of("exp07_real100_s2") == "exp07_real100"
    assert predict.data_of("exp07_real100") == "exp07_real100"


def test_rfdetr_command_copies_predictions_to_the_expected_name():
    cmd = predict.rfdetr_cmd("exp07_real100_s3", "val", "/runs/predictions/rfdetr/val", "cpu")
    assert cmd[0] == "sh"
    assert "/kit/prepare/out/rfdetr/exp07_real100 " in cmd[2]
    assert cmd[2].endswith("/runs/predictions/rfdetr/val/exp07_real100_s3_val_preds.json")


def test_diffusiondet_command_sets_steps_and_device():
    cmd = predict.diffusiondet_cmd("exp07_real100", "val", "/runs/p", "cpu", 2)
    assert "MODEL.DEVICE cpu" in cmd[2] and "MODEL.DiffusionDet.SAMPLE_STEP 2" in cmd[2]
    assert "densea_valid" in cmd[2]


def test_test_split_requires_confirmation(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    out = subprocess.run([sys.executable, str(RUN / "predict.py"), "--detector", "rtdetrv2", "--split", "test",
                          "--exps", "exp07_real100", "--dry-run"], capture_output=True, text=True)
    assert out.returncode != 0 and "ALLOW_TEST" in (out.stderr + out.stdout)


def test_dry_run_prints_the_container_command(tmp_path):
    out = subprocess.run([sys.executable, str(RUN / "predict.py"), "--detector", "rtdetrv2", "--split", "val",
                          "--exps", "exp07_real100", "--dry-run"], capture_output=True, text=True,
                         env={**os.environ, "RUNS_DIR": str(tmp_path)})
    assert out.returncode == 0
    assert "docker exec" in out.stdout and "infer_val_cpu.py" in out.stdout


def test_guard_reports_failure_when_the_process_fails(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    assert run_phase.wait_with_guard(proc, "none", None) is False


@pytest.mark.parametrize("phase,detector", [(4, "rfdetr")])
def test_phase4_is_restricted_to_rtdetrv2(phase, detector):
    out = subprocess.run([sys.executable, str(RUN / "run_phase.py"), "--phase", str(phase), "--detector", detector],
                         capture_output=True, text=True)
    assert out.returncode != 0
