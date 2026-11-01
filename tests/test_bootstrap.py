"""Group bootstrap and evaluation helpers (analysis/proxy, analysis/full_val, eval/coco_tools) on a toy COCO set."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
sys.path.insert(0, str(KIT / "analysis/full_val"))
from common import paths  # noqa: E402
from eval.coco_tools import Cache, ap_batch_cpu, load_dets, load_gt, official  # noqa: E402
import full_val_bootstrap as FV  # noqa: E402
import px_val_bootstrap as PX  # noqa: E402
import dd_backbones_bootstrap as DB  # noqa: E402
import series_bootstrap as SB  # noqa: E402
import test_analysis as TA  # noqa: E402

N_IMG, N_GROUPS = 12, 4


def toy_gt(path):
    """12 images in 4 groups; each image has two objects of different classes."""
    images = [{"id": i, "file_name": f"images/x/{i}.jpg", "width": 200, "height": 200, "group": i // 3}
              for i in range(N_IMG)]
    anns = []
    for i in range(N_IMG):
        for k, (x, cat) in enumerate(((10, i % 8), (100, (i + 3) % 8))):
            anns.append({"id": 2 * i + k + 1, "image_id": i, "category_id": cat, "bbox": [x, 20, 60, 50],
                         "area": 3000, "iscrowd": 0})
    cats = [{"id": c, "name": n} for c, n in enumerate(paths.CLASSES)]
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(path, "w"))
    return anns


def toy_preds(path, anns, shift=0, drop=()):
    """Detections equal to the GT (optionally shifted); images in `drop` get none."""
    dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "score": 0.9 - 0.01 * a["id"] / 10,
             "bbox": [a["bbox"][0] + shift, *a["bbox"][1:]]} for a in anns if a["image_id"] not in drop]
    json.dump(dets, open(path, "w"))


@pytest.fixture
def toy(tmp_path):
    anns = toy_gt(tmp_path / "gt.json")
    groups = [{"group": g, "split": "val", "image_ids": [i for i in range(N_IMG) if i // 3 == g]}
              for g in range(N_GROUPS)]
    json.dump(groups, open(tmp_path / "groups.json", "w"))
    return tmp_path, anns


def test_group_weights_resample_whole_groups(toy):
    tmp, _ = toy
    w, n_groups = PX.group_weights(list(range(N_IMG)), 50, groups_file=tmp / "groups.json")
    assert n_groups == N_GROUPS and w.shape == (50, N_IMG)
    for g in range(N_GROUPS):                                     # images of a group share their weight
        block = w[:, 3 * g:3 * g + 3]
        assert (block == block[:, :1]).all()
    assert (w[:, ::3].sum(1) == N_GROUPS).all()                   # one draw per group
    w2, _ = PX.group_weights(list(range(N_IMG)), 50, groups_file=tmp / "groups.json")
    assert np.array_equal(w, w2)
    w3, _ = PX.group_weights(list(range(N_IMG)), 50, seed=5, groups_file=tmp / "groups.json")
    assert not np.array_equal(w, w3)


def test_group_weights_do_not_depend_on_group_labels_order(toy):
    tmp, _ = toy
    ids = [0, 3, 6, 9]
    w, n_groups = PX.group_weights(ids, 10, groups_file=tmp / "groups.json")
    counts = np.random.default_rng(21).multinomial(N_GROUPS, np.ones(N_GROUPS) / N_GROUPS, size=10)
    assert n_groups == N_GROUPS and np.array_equal(w, counts)


def test_diff_row_ci_and_p():
    db = np.linspace(1, 5, 4001)
    out = {}
    row = FV.diff_row("A − B", 3.0, db, "AP", out, "pairs")
    assert row == "| A − B | AP | +3.00 | [+1.10; +4.90] | 0.0000 |"
    assert out["pairs"][0]["ci"] == [1.1, 4.9] and out["pairs"][0]["p"] == 0.0
    both = FV.diff_row("A − B", 0.0, np.linspace(-1, 1, 1001), "AP", {}, "k")
    assert both.endswith("| 1.0000 |")


def test_evaluate_perfect_predictions(toy):
    tmp, anns = toy
    toy_preds(tmp / "exp_val_preds.json", anns)
    gt = load_gt(tmp / "gt.json")
    w, _ = PX.group_weights(sorted(gt.getImgIds()), 20, groups_file=tmp / "groups.json")
    res = PX.evaluate("exp", gt, tmp, w)
    assert res["official"]["AP"] == pytest.approx(1.0)
    assert res["AP0"] == pytest.approx(1.0) and np.allclose(res["APb"], 1.0)


def test_cache_reaccumulation_matches_official_on_subsets(toy):
    tmp, anns = toy
    toy_preds(tmp / "p.json", anns, shift=12, drop={5})
    gt = load_gt(tmp / "gt.json")
    dt = load_dets(tmp / "p.json", gt)
    cache = Cache(gt, dt)
    subset = [0, 1, 2, 3, 6, 7, 8]
    w = np.zeros((1, N_IMG))
    w[0, subset] = 1
    off = official(gt, dt, subset, paths.CLASSES)
    for met, t in (("AP", None), ("AP50", [0]), ("AP75", [5])):
        assert ap_batch_cpu(cache, w, t)[0][0] == pytest.approx(off[met], abs=1e-9)


def test_load_dets_rejects_unknown_ids(toy):
    tmp, _ = toy
    json.dump([{"image_id": 999, "category_id": 0, "bbox": [0, 0, 5, 5], "score": 0.5}], open(tmp / "bad.json", "w"))
    with pytest.raises(ValueError):
        load_dets(tmp / "bad.json", load_gt(tmp / "gt.json"))


def test_iou_and_same_class_matching():
    a = np.array([[0.0, 0, 10, 10]])
    b = np.array([[0.0, 0, 10, 10], [1, 0, 10, 10], [50, 50, 5, 5]])
    assert np.allclose(TA.iou_mat(a, b)[0], [1.0, 90 / 110, 0.0])
    gb, gc = b, np.array([1, 1, 2])
    db, dc = np.array([[0.0, 0, 10, 10], [0, 0, 10, 10]]), np.array([1, 1])
    assert list(TA.match_same_class(gb, gc, db, dc)) == [0, 1]        # the second detection takes the next free GT
    assert list(TA.match_same_class(gb, gc, db, np.array([2, 1]))) == [-1, 0]


def test_best_threshold_and_confusion_counts():
    gb = np.array([[0.0, 0, 10, 10], [50, 50, 10, 10]])
    gc = np.array([0, 1])
    db = np.array([[0.0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]])
    dc, ds = np.array([0, 0, 3]), np.array([0.9, 0.8, 0.3])
    per_img = {0: (gb, gc, db, dc, ds)}
    thr, f1, p, r = TA.best_threshold(per_img)
    assert thr == 0.9 and f1 == pytest.approx(2 / 3) and p == 1.0 and r == 0.5
    cm = TA.confusion(per_img, 0.5)
    assert cm[0, 0] == 1 and cm[1, 0] == 1 and cm.sum() == 2           # hit and class confusion; low-score FP dropped
    cm = TA.confusion(per_img, 0.0)
    assert cm[TA.K, 3] == 1                                            # background FP of class 3
    st = TA.cm_stats(cm)
    assert st["Bottle"]["TP"] == 1 and st["Can"]["FN_pct"] == 100.0


def test_series_hundred_percent_arms_use_three_seeds():
    arms = SB.SERIES[100]
    for key, tag in (("real", "exp07_real100"), ("cp", "exp06v11_synth100_real100"), ("v9_mf", "exp08_real_synth_minor")):
        assert arms[key] == [tag, tag + "_s2", tag + "_s3"]
    assert arms["v9"] == ["exp06_synth100_real100"]


def test_series_arm_is_mean_of_runs_with_same_resamples():
    res = {"a": {"AP0": 0.4, "APb": np.array([0.3, 0.5])}, "b": {"AP0": 0.5, "APb": np.array([0.5, 0.7])}}
    v0, vb = SB.arm(res, ["a", "b"], "AP")
    assert v0 == pytest.approx(45.0) and vb == pytest.approx([40.0, 60.0])


def test_dd_backbones_compares_each_backbone_with_resnet50(toy):
    tmp, anns = toy
    for sub, name, shift in (("steps1", "exp07_real100", 0), ("backbones", "res101", 30)):
        (tmp / sub).mkdir()
        toy_preds(tmp / sub / f"{name}_val_preds.json", anns, shift=shift)
    gt = load_gt(tmp / "gt.json")
    w, n_groups = PX.group_weights(sorted(gt.getImgIds()), 40, groups_file=tmp / "groups.json")
    md, out = DB.compare(tmp, gt, w, n_groups, 40)
    assert list(out["runs"]) == ["ResNet-50", "ResNet-101"]           # backbones without predictions are omitted
    assert out["runs"]["ResNet-50"]["AP"] == pytest.approx(100.0)
    assert [p["metric"] for p in out["pairs"]] == ["AP", "AP50", "AP75"]
    assert all(p["pair"] == "ResNet-101 − ResNet-50" and p["diff"] < 0 for p in out["pairs"])
    assert sum(line.startswith("| ResNet-101 − ResNet-50") for line in md) == 3
