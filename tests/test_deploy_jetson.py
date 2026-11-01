"""CPU-only tests of the pure helpers of deploy_jetson (no TensorRT, GPU, detectron2 or ModelOpt needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

KIT = Path(__file__).resolve().parents[1]
DEPLOY = KIT / "deploy_jetson"
for p in (KIT, DEPLOY, DEPLOY / "jetson"):
    sys.path.insert(0, str(p))

import eval_rfdetr_val  # noqa: E402
import export_onnx_diffusiondet as dd_export  # noqa: E402
import export_onnx_rtdetrv2 as rt_export  # noqa: E402
import int8_bootstrap  # noqa: E402
import latency_5090  # noqa: E402
import make_calibration_set as mcs  # noqa: E402
import power  # noqa: E402
import prof_regions  # noqa: E402
import rtdetrv2_io  # noqa: E402
from eval_utils import time_stats  # noqa: E402


# ---------------------------------------------------------------- calibration set
def synthetic_train():
    """12 images from two sources; the classes A and B are common, C only appears in two images."""
    cats = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}, {"id": 2, "name": "C"}]
    images, anns = [], []
    for i in range(12):
        source = "S1" if i % 3 else "S2"
        images.append({"id": 100 - i, "file_name": f"images/{source}/{i}.jpg", "source": source, "group": i})
        anns.append({"image_id": 100 - i, "category_id": 0 if i % 2 else 1})
    anns += [{"image_id": 100 - 4, "category_id": 2}, {"image_id": 100 - 9, "category_id": 2}]
    return {"images": images, "annotations": anns, "categories": cats}


def test_index_sorts_images_and_names_classes():
    imgs, cats = mcs.index(synthetic_train())
    assert [im["id"] for im in imgs] == sorted(im["id"] for im in imgs)
    assert cats[96] == ["B", "C"]


def test_greedy_order_is_deterministic_and_nested():
    imgs, cats = mcs.index(synthetic_train())
    a = [im["id"] for im in mcs.greedy_order(imgs, cats, 8, seed=21)]
    assert a == [im["id"] for im in mcs.greedy_order(imgs, cats, 8, seed=21)]
    assert a[:4] == [im["id"] for im in mcs.greedy_order(imgs, cats, 4, seed=21)]
    assert len(set(a)) == 8


def test_greedy_order_takes_the_rare_class_first():
    imgs, cats = mcs.index(synthetic_train())
    first = mcs.greedy_order(imgs, cats, 3, seed=21)
    assert any("C" in cats[im["id"]] for im in first)


def test_check_disjoint_rejects_shared_groups():
    imgs, _ = mcs.index(synthetic_train())
    held = [{"group": imgs[0]["group"], "file_name": "images/S9/other.jpg"}]
    with pytest.raises(SystemExit):
        mcs.check_disjoint(imgs[:3], held)
    mcs.check_disjoint(imgs[:3], [{"group": 999, "file_name": "images/S9/other.jpg"}])


def test_shipped_calibration_list_is_train_ids_only():
    meta = json.loads((DEPLOY / "calibration" / "calib_100.json").read_text(encoding="utf8"))
    assert meta["split"] == "train" and meta["n"] == len(meta["images"]) == 100
    assert len({r["id"] for r in meta["images"]}) == 100
    assert set(meta["images"][0]) == {"id", "group", "source", "classes"}
    assert set(meta) == {"split", "seed", "n", "nested", "images"}


# ---------------------------------------------------------------- RT-DETRv2 export and I/O
def test_fusion_candidates_are_conv_norm_pairs():
    pairs = rt_export.fusion_candidates()
    assert len(pairs) == 69 and len({tuple(p) for p in pairs}) == 69
    assert all(c.endswith(".conv") and n.endswith(".norm") and c[:-5] == n[:-5] for c, n in pairs)
    assert pairs[0] == ["backbone.conv1.conv1_1.conv", "backbone.conv1.conv1_1.norm"]
    assert pairs[3] == ["backbone.res_layers.0.blocks.0.short.conv", "backbone.res_layers.0.blocks.0.short.norm"]
    assert pairs[-1] == ["encoder.pan_blocks.1.conv3.conv", "encoder.pan_blocks.1.conv3.norm"]


def test_preprocess_shape_range_and_layout():
    for shape in ((480, 640, 3), (1200, 900, 3)):
        x = rtdetrv2_io.preprocess(np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8))
        assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
        assert 0.0 <= x.min() and x.max() <= 1.0


def test_preprocess_converts_bgr_to_rgb():
    bgr = np.zeros((640, 640, 3), np.uint8)
    bgr[..., 0] = 255   # blue channel in BGR
    x = rtdetrv2_io.preprocess(bgr)
    assert x[0, 2].min() == 1.0 and x[0, 0].max() == 0.0


def test_detections_keep_scores_above_threshold_as_xywh():
    labels, scores = np.array([1, 2]), np.array([0.5, 0.0005])
    boxes = np.array([[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 1.0, 1.0]])
    out = rtdetrv2_io.detections(7, labels, boxes, scores)
    assert out == [{"image_id": 7, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.5}]


# ---------------------------------------------------------------- RF-DETR and DiffusionDet pre/post-processing
def test_rfdetr_prepare_is_normalized_and_contiguous():
    from PIL import Image
    x = eval_rfdetr_val.prepare(Image.new("RGB", (90, 50), (124, 116, 104)), res=56)
    assert x.shape == (1, 3, 56, 56) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.abs(x).max() < 0.05   # the ImageNet mean maps to ~0


def test_rfdetr_postprocess_topk_and_box_conversion():
    dets = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]], np.float32)
    logits = np.full((3, 4), -10.0, np.float32)
    logits[1, 2] = 5.0
    logits[0, 3] = 3.0
    scores, labels, boxes = eval_rfdetr_val.postprocess(dets, logits, np.array([100.0, 200.0]), num_select=2)
    assert list(labels) == [2, 3] and scores[0] > scores[1]
    assert np.allclose(boxes[1], [40.0, 60.0, 60.0, 140.0])   # cxcywh (0.5, 0.5, 0.2, 0.4) on a 100 x 200 image


def test_diffusiondet_postprocess_applies_nms_rescale_and_drops_empty_boxes():
    logits = np.full((1, 3, 2), -8.0, np.float32)
    logits[0, 0, 0], logits[0, 1, 0], logits[0, 2, 1] = 4.0, 3.0, 2.0
    boxes = np.array([[[0, 0, 100, 100], [2, 2, 98, 98], [50, 50, 50, 50]]], np.float32)   # 0 and 1 overlap
    b, s, lab = dd_export.postprocess(logits, boxes, (100, 100), (200, 300), k=3)
    assert len(s) == 1 and lab[0] == 0 and s[0] == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-5)
    assert np.allclose(b[0], [0, 0, 300, 200])   # rescaled to (H0, W0) = (200, 300)


# ---------------------------------------------------------------- measurement helpers
TEGRA = ["RAM 1/7 VDD_IN 6000mW/6100mW VDD_CPU_GPU_CV 2000mW/2000mW VDD_SOC 1500mW/1500mW",
         "RAM 1/7 VDD_IN 7000mW/6500mW VDD_CPU_GPU_CV 3000mW/2500mW VDD_SOC 1500mW/1500mW"]


def test_power_rail_means_and_energy():
    assert power.rail_means(TEGRA) == {"VDD_IN": 6.5, "VDD_CPU_GPU_CV": 2.5, "VDD_SOC": 1.5, "samples": 2}
    out = power.summarize(TEGRA, "... Throughput: 26 qps\n")
    assert out["qps"] == 26.0 and out["mJ_per_inference_board"] == 250.0


def test_time_stats_discard_warm_up():
    s = time_stats([9.0, 9.0, 0.001, 0.002, 0.003], skip=2)
    assert s["mean"] == pytest.approx(2.0) and s["median"] == pytest.approx(2.0)


LAYERS = [{"Name": "/backbone/conv", "LayerType": "Convolution", "Outputs": [{"Format/Datatype": "Int8 format"}]},
          {"Name": "/decoder/layers.0/cross_attn", "LayerType": "Myelin", "Outputs": [{"Format/Datatype": "Half"}]},
          {"Name": "Reformatting CopyNode", "LayerType": "Reformat", "Outputs": []},
          {"Name": "other_layer", "LayerType": "Scale", "Outputs": [{"Format/Datatype": "Float"}]}]


def test_profile_regions_and_precisions():
    rules = prof_regions.parse_rules("backbone=/backbone/;attention=cross_attn")
    profile = {"/backbone/conv": 2.0, "/decoder/layers.0/cross_attn": 5.0, "Reformatting CopyNode": 0.5, "other_layer": 1.0}
    total, blocks, precs = prof_regions.split_times(LAYERS, profile, rules)
    assert total == 8.5
    assert dict(blocks) == {"backbone": 2.0, "attention": 5.0, "reformat": 0.5, "other": 1.0}
    assert dict(precs) == {"INT8": 2.0, "FP16": 5.0, "other": 0.5, "FP32": 1.0}


def test_latency_report_tables():
    cfg = {"net_ms": {"mean": 10.0, "median": 9.0, "p95": 12.0}, "e2e_ms": {"mean": 20.0, "median": 19.0, "p95": 25.0},
           "power": {"power_W": 200.0, "iter_s": 0.01, "J_img_gross": 2.0, "J_img_net": 1.0, "temp_C": 50.0,
                     "sm_clock_MHz": 2800.0}}
    raw = {"rtdetrv2": {"configs": {"fp32/no_grad": cfg}, "idle": {"power_W": 70.0}}}
    tables = "\n".join(latency_5090.latency_tables(raw) + latency_5090.power_table(raw))
    assert "| RT-DETRv2 | fp32/no_grad | 10.00 | 9.00 | 12.00 | 100.0 |" in tables
    assert "| RT-DETRv2 | fp32/no_grad | 200 | 70 | 10.00 | 2.00 | 1.00 | 50 | 2800 |" in tables


# ---------------------------------------------------------------- bootstrap sets
@pytest.mark.parametrize("name", list(int8_bootstrap.SETS))
def test_bootstrap_pairs_refer_to_declared_runs(name):
    title, labels, pairs = int8_bootstrap.SETS[name]
    assert title and all(a in labels and b in labels and a != b for a, b in pairs)
