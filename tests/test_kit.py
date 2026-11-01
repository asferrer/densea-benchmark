"""Unit tests of the kit (no GPU, no dataset)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_group.grouping import build_groups, cleansea_sequence, name_key  # noqa: E402
from split_group.group_stratify import group_matrix, split_groups  # noqa: E402
from prepare import strategies as st  # noqa: E402


def test_name_key_roboflow_variants():
    base = name_key("train_image_0417_jpg.rf.0123456789abcdef0123456789abcdef.jpg", "Ocean_garbage")
    assert name_key("a0_a1_train_image_0417_jpg.rf.ffffffffffffffffffffffffffffffff.jpg", "Ocean_garbage") == base
    assert name_key("uwg_g-153__flipv_jpg.rf.0123456789abcdef0123456789abcdef.jpg", "Neural_Ocean") == \
        name_key("uwg_g-153_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg", "Neural_Ocean")


def test_name_key_uses_only_the_base_name():
    a = name_key("images/Ocean_garbage/train/x_jpg.rf." + "0" * 32 + ".jpg", "Ocean_garbage")
    b = name_key("images/Ocean_garbage/test/x_jpg.rf." + "1" * 32 + ".jpg", "Ocean_garbage")
    assert a == b
    cs = "images/CleanSea/CocoFormatDataset/train_coco/JPEGImages/HPD2027OUT0109.jpg"
    assert name_key(cs, "CleanSea") == "CS:HPD2027OUT0109.jpg"


def test_cleansea_sequence_modes():
    a = "images/CleanSea/JPEGImages/HPD2042HDTV10014.jpg"
    b = "images/CleanSea/JPEGImages/HPD2042HDTV20014.jpg"
    assert cleansea_sequence(a, "dive") == cleansea_sequence(b, "dive")
    assert cleansea_sequence(a, "dive_camera") != cleansea_sequence(b, "dive_camera")


def test_groups_union_name_content_sequence():
    rows = [{"file_name": "x_jpg.rf." + "0" * 32 + ".jpg", "source": "Ocean_garbage"},
            {"file_name": "a0_x_jpg.rf." + "1" * 32 + ".jpg", "source": "Ocean_garbage"},
            {"file_name": "y.jpg", "source": "Neural_Ocean"},
            {"file_name": "JPEGImages/HPD0001OUT0001.jpg", "source": "CleanSea"},
            {"file_name": "JPEGImages/HPD0001HDTV0002.jpg", "source": "CleanSea"},
            {"file_name": "z.jpg", "source": "Neural_Ocean"}]
    groups, stats, _, _ = build_groups(rows, [(1, 2, 10, 0.99)])
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3, 4], [5]]


def test_split_groups_keeps_groups_and_ratios():
    rng = np.random.default_rng(0)
    groups = [[i] for i in range(400)]
    cls = [[int(c)] for c in rng.integers(0, 3, 400)]
    m = group_matrix(groups, cls, ["A"] * 400, ["A"], 3)
    assign, cur, _, _ = split_groups(m, (0.7, 0.15, 0.15), 3, seed=1, restarts=20)
    frac = cur[:, -1] / 400
    assert np.allclose(frac, [0.7, 0.15, 0.15], atol=0.02)


def _toy(n_img, cls_of):
    ims = [{"id": i, "file_name": f"{i}.jpg"} for i in range(n_img)]
    anns = [{"id": i, "image_id": i, "category_id": cls_of(i), "bbox": [0, 0, 1, 1]} for i in range(n_img)]
    return {"images": ims, "annotations": anns, "categories": [{"id": 0, "name": "Can"}, {"id": 1, "name": "Tire"}]}


def test_strategies_deficits_and_merge():
    real = _toy(100, lambda i: 0 if i < 10 else 1)              # Can 10, Tire 90
    synth = _toy(500, lambda i: i % 2)
    part, log = st.minority_focus(real, synth, seed=21, target=50, classes=["Can"])
    assert log["Can"]["needed"] == 40 and log["Can"]["added"] == 40
    part2, log2 = st.inverse_proportional(real, synth, seed=21, target=100)
    assert log2["Can"]["needed"] == 90 and log2["Tire"]["needed"] == 10
    merged = st.merge(part, real, "t")
    ids = [im["id"] for im in merged["images"]]
    assert len(ids) == len(set(ids)) == len(part["images"]) + 100
    assert {im["source"] for im in merged["images"]} == {"real", "synthetic"}


def test_strategies_count_only_pasted_objects():
    """Copy-paste batch: real background objects (origin real) neither cover a deficit nor make an image a candidate."""
    real = _toy(100, lambda i: 0 if i < 10 else 1)              # Can 10, Tire 90
    synth = _toy(200, lambda i: 0)                              # 200 images with one Can each
    for a in synth["annotations"]:
        a["origin"] = "real" if a["image_id"] < 100 else "pasted"
    part, log = st.minority_focus(real, synth, seed=21, target=50, classes=["Can"])
    assert log["Can"]["added"] == 40 and len(part["images"]) == 40
    assert all(im["id"] >= 100 for im in part["images"])
    _, log2 = st.inverse_proportional(real, synth, seed=21, target=500)
    assert log2["Can"]["needed"] == 490 and log2["Can"]["added"] == 100   # only 100 pasted objects exist
    assert st.is_pasted({}) and not st.is_pasted({"origin": "real"})


def test_stratified_subset_deterministic():
    real = _toy(1000, lambda i: i % 2)
    a = {im["id"] for im in st.stratified_subset(real, 0.25, 21)["images"]}
    b = {im["id"] for im in st.stratified_subset(real, 0.25, 21)["images"]}
    assert a == b and len(a) == 250


def test_coco_tools_bootstrap_matches_official():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from eval.coco_tools import Cache, ap_batch_cpu, official
    gt = COCO()
    gt.dataset = {"images": [{"id": i, "width": 100, "height": 100} for i in range(20)],
                  "categories": [{"id": 0, "name": "Bottle"}, {"id": 1, "name": "Can"}],
                  "annotations": [{"id": i, "image_id": i, "category_id": i % 2, "bbox": [10, 10, 30, 30],
                                   "area": 900, "iscrowd": 0} for i in range(20)]}
    gt.createIndex()
    rng = np.random.default_rng(0)
    dets = [{"image_id": i, "category_id": i % 2, "score": float(rng.random()),
             "bbox": [10 + float(rng.integers(0, 15)), 10, 30, 30]} for i in range(20)]
    dt = gt.loadRes(dets)
    off = official(gt, dt, list(range(20)), ["Bottle", "Can"])
    ap0 = ap_batch_cpu(Cache(gt, dt), np.ones((1, 20)))[0][0]
    assert abs(ap0 - off["AP"]) < 1e-9
