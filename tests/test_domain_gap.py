"""Tests of domain_gap with toy data (no real images, no network)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import domain_gap as dg  # noqa: E402


def test_allocate_sums_to_n_and_respects_the_minimum():
    q = dg.allocate({"a": 900, "b": 90, "c": 10}, 100, min_q=20)
    assert sum(q.values()) == 100
    assert q["c"] == 10 and q["b"] >= 20 and q["a"] >= 20


def test_stratified_sample_is_deterministic():
    items = [{"s": "x", "i": i} for i in range(50)] + [{"s": "y", "i": i} for i in range(50)]
    a = dg.stratified_sample(items, lambda d: d["s"], {"x": 5, "y": 7})
    b = dg.stratified_sample(items, lambda d: d["s"], {"x": 5, "y": 7})
    assert a == b and len(a) == 12


def test_expand_box_clips_to_the_image():
    assert dg.expand_box([10, 10, 100, 50], 200, 100, 0.2) == (0, 0, 130, 70)
    assert dg.expand_box([0, 0, 10, 10], 100, 100, 0.2) == (0, 0, 12, 12)


def test_frechet_and_kid_distinguish_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (300, 16))
    b = rng.normal(0, 1, (300, 16))
    c = rng.normal(2, 1, (300, 16))
    assert dg.frechet_distance(a, a) < 1e-6
    assert dg.frechet_distance(a, c) > 10 * dg.frechet_distance(a, b)
    assert dg.kid(a, c)[0] > dg.kid(a, b)[0]


def test_cv_classify_auc_control_and_separable():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (200, 32))
    b = rng.normal(0, 1, (200, 32))
    c = rng.normal(1.5, 1, (200, 32))
    x0, y0 = dg._xy(a, b)
    x1, y1 = dg._xy(a, c)
    assert abs(dg.cv_classify(x0, y0)["auc"] - 0.5) < 0.12
    r = dg.cv_classify(x1, y1)
    assert r["auc"] > 0.95 and r["oof"].shape == (400,)
    g = dg.subgroup_auc(r["oof"], y1, ["u", "v"] * 200)
    assert set(g) == {"u", "v"}
