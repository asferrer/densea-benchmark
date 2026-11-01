"""Box bias (analysis/box_bias): greedy matching, bias per side, bootstrap and correction."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.box_bias import box_bias_core as C  # noqa: E402


def test_iou_identical_and_disjoint_boxes():
    m = C.iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 20]])
    assert np.allclose(m[0], [1.0, 0.0, 0.5])


def test_greedy_prioritises_score_and_respects_class():
    g = [[0, 0, 10, 10]]
    # two predictions on the same GT: the higher score wins (the worse box), the other has no pair
    r = C.greedy_match([[0, 0, 10, 10], [0, 0, 10, 8]], [1, 1], [0.4, 0.9], g, [1])
    assert [(i, j) for i, j, _ in r] == [(1, 0)]
    # different class: no pair
    assert C.greedy_match([[0, 0, 10, 10]], [2], [0.9], g, [1]) == []
    # IoU < 0.5: no pair
    assert C.greedy_match([[0, 0, 10, 4]], [1], [0.9], g, [1]) == []


def test_greedy_each_gt_once():
    r = C.greedy_match([[0, 0, 10, 10], [1, 1, 11, 11]], [0, 0], [0.9, 0.8],
                       [[0, 0, 10, 10], [1, 1, 11, 11]], [0, 0])
    assert sorted(j for _, j, _ in r) == [0, 1]


def test_bias_of_a_box_inflated_20px_per_side():
    g = np.array([100.0, 100.0, 140.0, 180.0])           # 40 x 80
    p = g + np.array([-20, -20, 20, 20])                  # +20 px per side
    b = C.side_bias(p, g)
    assert (b["out_l"], b["out_t"], b["out_r"], b["out_b"]) == (20, 20, 20, 20)
    assert np.isclose(b["ratio_w"], 80 / 40) and np.isclose(b["ratio_h"], 120 / 80)
    assert np.isclose(b["ratio_area"], (80 * 120) / (40 * 80))
    assert np.isclose(b["rel_l"], 20 / 40) and np.isclose(b["rel_t"], 20 / 80)


def test_tight_box_has_no_bias():
    g = np.array([5.0, 5.0, 25.0, 45.0])
    b = C.side_bias(g, g)
    assert b["ratio_area"] == 1 and b["out_l"] == b["out_b"] == 0


def test_group_bootstrap_ci_contains_the_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    grp = np.repeat(np.arange(30), 5)
    val = rng.normal(2.0, 0.1, 150)
    cnt = C.boot_counts(30, 500, 1)
    s1, s2 = C.summarize(val, grp, cnt), C.summarize(val, grp, C.boot_counts(30, 500, 1))
    assert s1 == s2 and s1["mean_ci"][0] < s1["mean"] < s1["mean_ci"][1]
    assert s1["median_ci"][0] < 2.0 < s1["median_ci"][1]


def test_weighted_median_with_known_resamples():
    val, grp = np.array([1.0, 2.0, 10.0]), np.array([0, 1, 2])
    cnt = np.array([[1, 1, 1], [0, 0, 3]])
    assert list(C.weighted_stat(val, grp, cnt, "median")) == [2.0, 10.0]
    assert np.allclose(C.weighted_stat(val, grp, cnt, "mean"), [13 / 3, 10.0])


def test_px_correction_reverts_the_inflation():
    gt = np.array([[100.0, 100, 140, 180], [10, 20, 50, 60]])
    pred = gt + np.array([-20, -20, 20, 20])
    rows = {"gw": gt[:, 2] - gt[:, 0], "gh": gt[:, 3] - gt[:, 1], "out_l": np.full(2, 20.0),
            "out_t": np.full(2, 20.0), "out_r": np.full(2, 20.0), "out_b": np.full(2, 20.0)}
    sh = C.fit_shift(rows, "px")
    assert np.allclose(C.shrink_boxes(pred, sh, "px"), gt)


def test_relative_correction_reverts_proportional_inflation():
    gt = np.array([[0.0, 0, 100, 100], [0, 0, 40, 40]])
    pred = np.array([[-10.0, -10, 110, 110], [-4, -4, 44, 44]])           # +10 % of the GT per side
    rows = {"gw": np.array([100.0, 40]), "gh": np.array([100.0, 40]), "out_l": np.array([10.0, 4]),
            "out_t": np.array([10.0, 4]), "out_r": np.array([10.0, 4]), "out_b": np.array([10.0, 4])}
    sh = C.fit_shift(rows, "rel")
    assert np.allclose(C.shrink_boxes(pred, sh, "rel"), gt)


def test_shrink_does_not_collapse_tiny_boxes():
    b = np.array([[0.0, 0, 5, 5]])
    assert np.allclose(C.shrink_boxes(b, {"l": 3, "t": 3, "r": 3, "b": 3}, "px"), b)
