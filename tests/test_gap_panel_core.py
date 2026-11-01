"""Tests of gap_panel_core with synthetic toy data (no images, no network, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import gap_panel_core as gc  # noqa: E402


def _gauss(n, d=16, shift=0.0, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)) + shift


def test_frechet_fast_matches_domain_gap():
    import domain_gap as dg
    a, b = _gauss(120, 12, 0, 1), _gauss(150, 12, 0.7, 2)
    assert abs(gc.frechet_fast(a, b) - dg.frechet_distance(a, b)) < 1e-6 * (1 + dg.frechet_distance(a, b))


def test_frechet_is_zero_with_itself():
    a = _gauss(80)
    assert abs(gc.frechet_fast(a, a)) < 1e-8


def test_unbiased_kid_is_near_zero_for_the_same_distribution_and_grows_with_a_shift():
    vals = [gc.kid_unbiased(_gauss(200, seed=s), _gauss(200, seed=100 + s)) for s in range(10)]
    assert abs(np.mean(vals)) < 5e-3
    assert gc.kid_unbiased(_gauss(200, seed=1), _gauss(200, 16, 1.0, 2)) > 10 * abs(np.mean(vals)) + 1e-3


def test_biased_cmmd_is_non_negative_and_zero_with_itself():
    a = _gauss(60, 8)
    assert abs(gc.cmmd(a, a)) < 1e-9
    assert gc.cmmd(_gauss(60, 8, 0, 1), _gauss(60, 8, 3.0, 2)) > 0


def test_cmmd_is_invariant_to_the_scale_of_the_embeddings():
    a, b = _gauss(50, 8, 0, 1), _gauss(50, 8, 1.5, 2)
    assert abs(gc.cmmd(a, b) - gc.cmmd(7.0 * a, 0.3 * b)) < 1e-9          # L2-normalized


def test_cmmd_known_value_two_points():
    # x = e1, y = e2 (unit vectors): |x-y|^2 = 2 -> k = exp(-2 / (2*100)); biased with n = 1 on each side
    x, y = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    k = np.exp(-2 / 200)
    assert abs(gc.cmmd(x, y) - 1000 * (1 + 1 - 2 * k)) < 1e-9


def test_prdc_same_distribution_and_disjoint_sets():
    r = _gauss(300, 8, 0, 1)
    same = gc.prdc(r, _gauss(300, 8, 0, 2))
    assert same["precision"] > 0.7 and same["recall"] > 0.7 and 0.6 < same["density"] < 1.6 and same["coverage"] > 0.7
    e1 = np.r_[6.0, np.zeros(7)]
    far = gc.prdc(_gauss(300, 8, 0, 1) + e1, _gauss(300, 8, 0, 2) - e1)           # opposite clusters
    assert far["precision"] < 0.05 and far["recall"] < 0.05 and far["coverage"] < 0.05


def test_prdc_collapsed_mode_lowers_recall_not_precision():
    r = _gauss(300, 8, 0, 1)
    fake = np.tile(r[:1], (300, 1)) + 1e-3 * _gauss(300, 8, 0, 3)     # everything falls on a single real point
    out = gc.prdc(r, fake)
    assert out["recall"] < 0.3                                              # covers little of the real set
    assert out["coverage"] < 0.3


def test_nn_to_real_exact_copies_give_zero():
    r = _gauss(50, 8)
    assert gc.nn_to_real(r, r.copy()) < 1e-9
    assert gc.nn_to_real(r, _gauss(50, 8, 0, 9)) > 0.05


def test_grouped_auc_separable_and_indistinguishable():
    gr = [f"g{i // 5}" for i in range(100)]
    gf = [f"s{i // 5}" for i in range(100)]
    cl = (["a", "b"] * 50)
    sep = gc.grouped_auc(_gauss(100, 8, 0, 1), _gauss(100, 8, 2.0, 2), gr, gf, cl, cl)
    same = gc.grouped_auc(_gauss(100, 8, 0, 3), _gauss(100, 8, 0, 4), gr, gf, cl, cl)
    assert sep["auc"] > 0.95 and same["auc"] < 0.7
    assert sep["auc_ci"][0] <= sep["auc"] <= sep["auc_ci"][1] + 1e-9 and set(sep["per_class"]) == {"a", "b"}


def test_tstr_separable_classes_and_absent_class():
    cls = ["a", "b", "c"]
    mu = {"a": 0.0, "b": 4.0, "c": -4.0}
    tr_y = np.array(cls * 40)
    tr_x = np.array([_gauss(1, 6, mu[c], i)[0] for i, c in enumerate(tr_y)])
    te_y = np.array(cls * 20)
    te_x = np.array([_gauss(1, 6, mu[c], 500 + i)[0] for i, c in enumerate(te_y)])
    out = gc.tstr(tr_x, tr_y, te_x, te_y, cls, test_groups=[f"g{i}" for i in range(len(te_y))])
    assert out["bacc"] > 0.95 and set(out["recall"]) == set(cls)
    sub = gc.tstr(tr_x, tr_y, te_x, te_y, ["a", "b"])          # same set of classes on both sides
    assert set(sub["recall"]) == {"a", "b"}


def test_group_subsample_ci_contains_the_point_value():
    a, b = _gauss(200, 8, 0, 1), _gauss(200, 8, 0.5, 2)
    ga, gb = [f"r{i // 4}" for i in range(200)], [f"s{i // 4}" for i in range(200)]
    ci = gc.group_subsample_ci(lambda x, y: {"kid": gc.kid_unbiased(x, y)}, a, b, ga, gb, b=15)
    lo, hi = ci["kid"]
    assert lo <= hi and lo - 0.05 < gc.kid_unbiased(a, b) < hi + 0.05


def test_pairwise_agreement_and_spearman():
    ap = {"v9": 35.0, "p": 36.6, "q": 36.1}
    good = {"v9": 0.9, "p": 0.5, "q": 0.6}                                  # AUC: higher = worse -> sense -1
    assert gc.pairwise_agreement(good, ap, sense=-1) == (3, 3)
    assert gc.pairwise_agreement(good, ap, sense=-1, min_ap_diff=1.0) == (2, 2)   # the p-q pair (dAP 0.5) is ignored
    bad = {"v9": 0.5, "p": 0.9, "q": 0.6}
    assert gc.pairwise_agreement(bad, ap, sense=-1)[0] < 3
    assert gc.spearman(good, ap, sense=-1) == pytest.approx(1.0)
    assert np.isnan(gc.spearman({"a": 1}, {"a": 1}))                         # n < 3: no correlation


def test_exact_minimum_spearman_p():
    assert gc.spearman_min_p(4) == pytest.approx(2 / 24) and gc.spearman_min_p(3) == pytest.approx(1 / 3)
    assert gc.spearman_min_p(6) == pytest.approx(2 / 720)


def test_rank_order_respects_the_sense():
    assert gc.rank_order({"a": 0.9, "b": 0.5}, sense=-1) == ["b", "a"]
    assert gc.rank_order({"a": 0.9, "b": 0.5}, sense=1) == ["a", "b"]
