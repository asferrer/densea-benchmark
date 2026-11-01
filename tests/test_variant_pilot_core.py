"""Tests of variant_pilot_core with toy features (no images, no network, no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import variant_pilot_core as av  # noqa: E402


def test_vendi_extremes():
    assert abs(av.vendi_score(np.ones((10, 8))) - 1.0) < 1e-6        # all equal -> 1
    assert abs(av.vendi_score(np.eye(6)) - 6.0) < 1e-6               # orthogonal -> n


def test_nn_distance_duplicates_and_orthogonal():
    x = np.array([[1.0, 0], [1, 0], [0, 1], [0, 1]])
    assert av.mean_nn_distance(x) < 1e-9                              # each one has a twin
    assert abs(av.mean_nn_distance(np.eye(5)) - 1.0) < 1e-9


def test_common_quota_minimum_and_cap():
    q = av.common_quota([{"Bottle": 300, "Can": 10}, {"Bottle": 200}], cap=125)
    assert q["Bottle"] == 125 and q["Can"] == 0 and q["Tire"] == 0


def test_pick_indices_deterministic_and_size():
    meta = [{"source": "a"}] * 70 + [{"source": "b"}] * 30
    i1 = av.pick_indices(meta, lambda m: m["source"], 20)
    assert len(i1) == 20 and (i1 == av.pick_indices(meta, lambda m: m["source"], 20)).all()
    assert (i1 < 70).sum() == 14


def test_analyze_variants_separates_diversity_and_realism():
    rng = np.random.default_rng(0)
    cls = av.dg.CLASSES
    quotas = {c: 12 for c in cls}
    meta = [{"cls": c} for c in cls for _ in range(12)]
    real = {"img": rng.normal(size=(60, 16)), "img_val": rng.normal(size=(60, 16)),
            "obj": rng.normal(size=(96, 16)), "obj_val": rng.normal(size=(96, 16)), "obj_meta": meta}
    clone = {"img": rng.normal(size=(60, 16)) + 3, "obj": np.stack([rng.normal(size=(8, 16))] * 12, axis=1).reshape(-1, 16) + 3,
             "obj_meta": meta}
    rep = av.analyze_variants({"v": clone}, real, quotas)
    assert rep["v"]["image"]["auc"] > 0.95                            # shifted: separable
    assert rep["real_ref"]["control_image"]["auc"] < 0.75             # real vs real: chance
    assert rep["v"]["object_diversity"]["mean"]["vendi"] < 2.0        # one clone per class
    assert rep["real_ref"]["object_diversity"]["mean"]["vendi"] > 5.0


def _bg_map(tmp_path):
    ann = {"images": [{"id": 5, "file_name": "00005_a.jpg"}, {"id": 9, "file_name": "00009_b.jpg"},
                      {"id": 77, "file_name": "00077_c.jpg"}]}
    f = tmp_path / "annotations.json"
    f.write_text(json.dumps(ann), encoding="utf8")
    # 5 and 9 share a group (near-duplicates); 77 is not in the partition (ignored)
    return av.background_group_map(f, gmap={5: "g1", 9: "g1"})


def test_background_group_map_uses_the_partition_group(tmp_path):
    assert _bg_map(tmp_path) == {"00005_a.jpg": "g1", "00009_b.jpg": "g1"}


def test_background_group_map_accepts_the_base_name(tmp_path):
    f = tmp_path / "annotations.json"
    f.write_text(json.dumps({"images": [{"id": 5, "file_name": "images/CleanSea/00005_a.jpg"}]}), encoding="utf8")
    m = av.background_group_map(f, gmap={5: "g7"})
    assert m["00005_a.jpg"] == m["images/CleanSea/00005_a.jpg"] == "g7"


def test_traceable_real_background_inherits_the_real_group(tmp_path):
    ims = [{"id": 0, "background": "00005_a.jpg"}, {"id": 1, "background": "00009_b.jpg"}]
    g, info = av.synth_image_groups("g_v11_copypaste", ims, _bg_map(tmp_path), real_bg=True)
    assert g == {0: "g1", 1: "g1"} and info["traceable"] == 2 and "partition group" in info["criterion"]


def test_untraceable_real_background_falls_back_to_per_image_group_and_warns(tmp_path):
    ims = [{"id": 0, "background": "zzz.jpg"}, {"id": 1}]
    g, info = av.synth_image_groups("i_v11_all", ims, _bg_map(tmp_path), real_bg=True)
    assert g[0] != g[1] and g[0].startswith("s:i_v11_all") and info["traceable"] == 0
    assert "NOT TRACEABLE" in info["criterion"]


def test_synthetic_without_real_background_one_group_per_background_or_per_image(tmp_path):
    bg = _bg_map(tmp_path)
    g, info = av.synth_image_groups("c", [{"id": 0, "background": "x.png"}, {"id": 1, "background": "x.png"},
                                          {"id": 2, "background": "y.png"}], bg, real_bg=False)
    assert g[0] == g[1] != g[2] and "per background" in info["criterion"]
    g, info = av.synth_image_groups("a_v9", [{"id": 0}, {"id": 1}], bg, real_bg=False)
    assert g[0] != g[1] and info["n_groups"] == 2 and "per synthetic image" in info["criterion"]


def test_attach_groups_objects_share_the_group_of_their_image(tmp_path):
    d = tmp_path / "g_v11_copypaste"
    (d / "images").mkdir(parents=True)
    (d / "synthetic_dataset.json").write_text(json.dumps({"images": [
        {"id": 0, "background": "00005_a.jpg"}, {"id": 1, "background": "nope.jpg"}]}), encoding="utf8")
    p = lambda i: str(d / "images" / f"s{i}.jpg")  # noqa: E731
    v = {"img_meta": [{"id": 0, "path": p(0)}, {"id": 1, "path": p(1)}],
         "obj_meta": [{"img": 0, "path": p(0)}, {"img": 0, "path": p(0)}, {"img": 1, "path": p(1)}]}
    old = av.REAL_BG_VARIANTS
    av.REAL_BG_VARIANTS = {"g_v11_copypaste"}
    try:
        av.attach_groups(v, "g_v11_copypaste", _bg_map(tmp_path))
    finally:
        av.REAL_BG_VARIANTS = old
    assert v["img_g"] == ["g1", "s:g_v11_copypaste:img1"]
    assert v["obj_g"][0] == v["obj_g"][1] == "g1" and v["obj_g"][2] == "s:g_v11_copypaste:img1"
    assert v["groups_info"]["traceable"] == 1 and "1 not traceable" in v["groups_info"]["criterion"]


def test_attach_groups_without_coco_falls_back_to_per_image(tmp_path):
    v = {"img_meta": [{"id": 3, "path": str(tmp_path / "x" / "images" / "a.jpg")}],
         "obj_meta": [{"img": 3, "path": str(tmp_path / "x" / "images" / "a.jpg")}]}
    av.attach_groups(v, "a_v9", {})
    assert v["img_g"] == v["obj_g"] == ["s:a_v9:img3"]


def test_grouped_auc_report_does_not_exploit_group_identity():
    rng = np.random.default_rng(0)
    # pure groups (each one all real or all synthetic) with the same distribution: there is no real domain signal,
    # but with ungrouped CV the classifier memorizes the group identity (high dimension, 60 centers)
    centers = rng.normal(size=(60, 200))
    x = np.stack([centers] * 5, axis=1).reshape(-1, centers.shape[1]) + rng.normal(size=(300, 200)) * 0.2
    g = [f"g{i // 5}" for i in range(300)]
    real, sint, gr, gs = x[:150], x[150:], g[:150], g[150:]
    r = av.auc_report(sint, real, gs, gr)
    assert r["auc"] > 0.75                       # group identity is exploited without grouping
    assert abs(r["auc_agr"] - 0.5) < 0.2         # grouped: chance
    assert "auc_agr" not in av.auc_report(sint, real)


def test_write_md_main_column_is_grouped(tmp_path):
    rep = {"real_ref": {"control_image": {"auc": .5, "auc_std": 0, "auc_agr": .52, "auc_agr_std": .01, "frechet": 1.0},
                        "control_object": {"auc": .5, "auc_std": 0, "auc_agr": .55, "auc_agr_std": .01, "frechet": 1.0},
                        "object_diversity": {}},
           "v": {"image": {"auc": .9, "auc_std": .01, "auc_agr": .8, "auc_agr_std": .02, "frechet": 2.0, "n_groups": 7},
                 "object": {"auc": .9, "auc_std": .01, "auc_agr": .7, "auc_agr_std": .02, "frechet": 2.0, "n_groups": 7},
                 "groups": {"criterion": "per image", "traceable": 0, "n_images": 3},
                 "object_diversity": {}}}
    p = tmp_path / "x.md"
    av.write_md(rep, {}, 3, p)
    t = p.read_text(encoding="utf8")
    assert "AUC img grouped | AUC img ungrouped" in t and "0.800 ± 0.020 | 0.900 ± 0.010" in t
    assert "Groups of the grouped CV" in t


def test_feature_cache_is_invalidated_when_the_sample_changes(tmp_path):
    dg = av.dg
    a = [{"path": "p1", "bbox": [0, 0, 1, 1]}, {"path": "p2", "bbox": [0, 0, 1, 1]}]
    dg._check_cache_items(tmp_path, a)
    (tmp_path / "c00000.npy").write_bytes(b"x")
    dg._check_cache_items(tmp_path, a)                        # same sample: kept
    assert (tmp_path / "c00000.npy").exists()
    dg._check_cache_items(tmp_path, a[:1])                    # different sample: stale chunks removed
    assert not (tmp_path / "c00000.npy").exists()
