"""Checkpoint selection of run/launch_diag.py (no GPU and no containers needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
import launch_diag as D  # noqa: E402


def touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def test_rtdetrv2_every_20_epochs_and_last(tmp_path):
    touch(tmp_path, "checkpoint0019.pth", "checkpoint0039.pth", "checkpoint0049.pth", "last.pth", "best.pth")
    found, _ = D.rt_ckpts(str(tmp_path))
    assert sorted(found) == ["ep020", "ep040", "last"]


def test_rfdetr_skips_10_epoch_checkpoints(tmp_path):
    touch(tmp_path, "checkpoint0009.pth", "checkpoint0019.pth", "checkpoint0029.pth", "checkpoint.pth",
          "checkpoint_best_ema.pth")
    found, _ = D.rf_ckpts(str(tmp_path))
    assert sorted(found) == ["ep020", "last"]


def test_diffusiondet_epochs_from_iterations(tmp_path):
    ipe = 7
    json.dump({"SOLVER.CHECKPOINT_PERIOD": 10 * ipe}, open(tmp_path / "schedule.json", "w"))
    touch(tmp_path, *(f"model_{e * ipe - 1:07d}.pth" for e in (20, 40, 50)), "model_best.pth")
    found, _ = D.dd_ckpts(str(tmp_path))
    assert found["ep020"].endswith(f"model_{20 * ipe - 1:07d}.pth")
    assert found["last"].endswith(f"model_{50 * ipe - 1:07d}.pth")
    assert sorted(found) == ["ep020", "ep040", "last"]


def test_diffusiondet_final_model_is_the_last(tmp_path):
    json.dump({"SOLVER.CHECKPOINT_PERIOD": 10}, open(tmp_path / "schedule.json", "w"))
    touch(tmp_path, "model_0000019.pth", "model_final.pth")
    found, _ = D.dd_ckpts(str(tmp_path))
    assert found == {"ep020": str(tmp_path / "model_0000019.pth"), "last": f"{tmp_path}/model_final.pth"}
