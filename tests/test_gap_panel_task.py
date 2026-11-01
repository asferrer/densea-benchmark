"""Tests of the task evaluator (RT-DETRv2 with 10 % real) with toy data: no model, no images, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import gap_extractors as gx  # noqa: E402
import gap_panel_task as gt  # noqa: E402


def test_registry_includes_the_task_evaluator_with_640_and_boxes():
    s = gx.REGISTRY[gt.MAIN]
    assert s.size == 640 and s.uses_rois and s.dim == 256 and s.mean_std == ([0.0] * 3, [1.0] * 3)
    assert gt.MAIN in gx.TASK and set(gx.TASK_SENS) <= set(gx.REGISTRY)
    assert gt.MAIN not in gx.ORDER and len(gx.ORDER) == 5          # the 5-extractor panel does not change
    assert gx.REGISTRY["rtdetrv2_real10_c5"].dim == 2048


def test_box_inside_the_crop_with_context_and_crop_at_the_border():
    c = gx.box_in_crop({"bbox": [100, 100, 100, 50], "w": 1000, "h": 1000})       # context 0.2 per side
    assert c == pytest.approx([20 / 140, 10 / 70, 120 / 140, 60 / 70], abs=0.02)
    b = gx.box_in_crop({"bbox": [0, 0, 100, 100], "w": 1000, "h": 1000})          # at the border: no context on the left
    assert b[0] == 0.0 and b[1] == 0.0 and 0.8 < b[2] < 0.9
    assert all(0.0 <= v <= 1.0 for v in gx.box_in_crop({"bbox": [900, 900, 300, 300], "w": 1000, "h": 1000}))


def test_roi_mean_averages_only_inside_the_box():
    torch = pytest.importorskip("torch")
    f = torch.zeros(1, 2, 10, 10)
    f[0, 0, 2:6, 3:7] = 5.0                      # cells y 2..5, x 3..6
    r = gx.roi_mean(f, torch.tensor([[0.3, 0.2, 0.7, 0.6]]))
    assert r.shape == (1, 2) and float(r[0, 0]) == pytest.approx(5.0) and float(r[0, 1]) == 0.0
    whole = gx.roi_mean(f, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert float(whole[0, 0]) == pytest.approx(5.0 * 16 / 100)


def test_roi_mean_tiny_box_uses_the_closest_cell():
    torch = pytest.importorskip("torch")
    f = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    r = gx.roi_mean(f, torch.tensor([[0.51, 0.51, 0.52, 0.52]]))          # between cell centers
    assert float(r[0, 0]) == pytest.approx(55.0)


def test_run_chunk_shares_the_tensor_and_passes_the_boxes(monkeypatch):
    torch = pytest.importorskip("torch")
    from PIL import Image
    seen = []

    def fn(x, rois):
        seen.append((tuple(x.shape), tuple(rois.shape)))
        return torch.zeros(len(x), 4)
    monkeypatch.setattr(gx, "_squash", lambda it, size=224: Image.new("RGB", (size, size)))
    monkeypatch.setattr(gx.REGISTRY["rtdetrv2_real10"], "_fn", fn)
    items = [{"bbox": [10, 10, 20, 20], "w": 100, "h": 100}] * 3
    out = gx._run_chunk(items, ["rtdetrv2_real10"])
    assert seen == [((3, 3, 640, 640), (3, 4))] and out["rtdetrv2_real10"].shape == (3, 4)


def test_ci_overlap_pairs_and_dispersion():
    auc = {"a": 0.5, "b": 0.7, "c": 0.9}
    ci = {"a": [0.45, 0.55], "b": [0.5, 0.9], "c": [0.85, 0.95]}
    assert gt.ci_overlap_pairs(auc, ci) == (1, 3)           # only a-c are disjoint
    assert gt.spread_over_noise(auc, None, ci) == pytest.approx(np.std([0.5, 0.7, 0.9]) / np.mean([0.05, 0.2, 0.05]))


def test_ranks_and_spearman_between_extractors():
    m = {"a": 1.0, "b": 2.0, "c": 3.0}
    r_auc = gt.rank_of(m, -1)               # high AUC = more gap -> best is a
    assert r_auc == {"a": 1, "b": 2, "c": 3}
    assert gt.spearman_ranks(r_auc, {"a": 1, "b": 2, "c": 3}) == pytest.approx(1.0)
    assert gt.spearman_ranks(r_auc, {"a": 3, "b": 2, "c": 1}) == pytest.approx(-1.0)


def test_separation_detects_disjoint_cis():
    sc = {"x": {"global": {"auc": 0.6, "auc_ci": [0.55, 0.65]}}, "y": {"global": {"auc": 0.8, "auc_ci": [0.75, 0.85]}},
          "z": {"global": {"auc": 0.62, "auc_ci": [0.5, 0.7]}}}
    s = gt.separation(sc, "x", "y")
    assert s["dAUC"] == pytest.approx(0.2) and s["disjoint"] and s["noise"] == pytest.approx(np.hypot(0.05, 0.05))
    assert not gt.separation(sc, "x", "z")["disjoint"]


def test_verdict_with_six_extractors_uses_the_adoption_rule():
    import gap_panel_drop as gd
    g = lambda auc, hi, v=10.0, cov=0.5: {"auc": auc, "auc_ci": [auc - 0.02, hi], "vendi": v, "coverage@val": cov}  # noqa: E731
    better = {"control": g(0.8, 0.82), "test": g(0.79, 0.81)}
    worse = {"control": g(0.8, 0.82), "test": g(0.9, 0.92)}
    res = {f"e{i}": better for i in range(3)}
    res.update({f"f{i}": worse for i in range(3)})
    v = gd.verdict(res)
    assert v["adopt"] and v["extractors_not_worse"] == 3 and v["n_extractors"] == 6
    res["f0"] = worse
    res["e0"] = worse
    assert not gd.verdict(res)["adopt"]
