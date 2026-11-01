"""Tests of variant_measures: sharpness, border and scale of the pasted objects on a minimal synthetic batch (CPU)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import variant_measures as m  # noqa: E402

REFERENCE = {"n": 100, "hf_ratio": 0.03, "lapvar64": 190.0, "noise_flat": 4.9, "border": 0.29}


def _variant(tmp_path):
    """One 200x200 noise image: a sharp pasted box at the border, a smooth one inside and a kept real one."""
    rng = np.random.default_rng(0)
    img = np.clip(120 + rng.normal(0, 30, (200, 200, 3)), 0, 255).astype(np.uint8)
    img[100:180, 100:180] = cv2.GaussianBlur(img[100:180, 100:180], (0, 0), 3.0)      # smooth box
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.png"), img)
    anns = [{"id": 1, "image_id": 0, "category_id": 0, "bbox": [0, 20, 60, 60], "origin": "pasted", "scale": 0.4},
            {"id": 2, "image_id": 0, "category_id": 0, "bbox": [100, 100, 80, 80], "origin": "pasted", "scale": 1.0},
            {"id": 3, "image_id": 0, "category_id": 0, "bbox": [60, 100, 30, 30], "origin": "real"},
            {"id": 4, "image_id": 0, "category_id": 0, "bbox": [60, 60, 8, 8], "origin": "pasted", "scale": 2.0}]
    d = {"images": [{"id": 0, "file_name": "a.png", "width": 200, "height": 200}],
         "categories": [{"id": 0, "name": "Bottle"}], "annotations": anns}
    (tmp_path / "synthetic_dataset.json").write_text(json.dumps(d), encoding="utf8")
    return tmp_path


def test_measure_variant_separates_pasted_and_real_and_counts_border(tmp_path):
    res = m.measure_variant(_variant(tmp_path))
    assert res["pasted"]["n"] == 2 and res["kept_real"]["n"] == 1     # the 8 px box is not measured
    assert res["pasted"]["border"] == 0.5
    assert res["scale"]["n"] == 2 and res["scale"]["bands"]["< 0.5"] == 0.5
    assert res["scale"]["bands"]["0.6 - 1.5"] == 0.5


def test_sharp_box_has_more_hf_than_smooth_box(tmp_path):
    root = _variant(tmp_path)
    img = cv2.imread(str(root / "images" / "a.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sharp, smooth = m._box_stats(img, (0, 20, 60, 60)), m._box_stats(img, (100, 100, 80, 80))
    assert sharp["hf_ratio"] > 3 * smooth["hf_ratio"] and sharp["lapvar64"] > smooth["lapvar64"]
    assert m._box_stats(img, (10, 10, 8, 8)) is None
    assert sharp["touch"] and not smooth["touch"]


def test_touches_border_tolerance():
    assert m.touches_border((3, 50, 60, 90), (200, 200)) and not m.touches_border((4, 50, 60, 90), (200, 200))
    assert m.touches_border((50, 50, 197, 90), (200, 200)) and not m.touches_border((50, 50, 196, 90), (200, 200))


def test_flat_noise_needs_enough_blocks():
    assert m.flat_noise(np.zeros((8, 8), np.float32)) is None
    assert m.flat_noise(np.zeros((40, 40), np.float32)) == 0.0


def test_scale_summary_bands_and_empty():
    s = m.scale_summary([0.3, 0.7, 1.0, 2.0])
    assert s["bands"] == {"< 0.5": 0.25, "0.5 - 0.6": 0.0, "0.6 - 1.5": 0.5, "> 1.5": 0.25}
    assert m.scale_summary([]) == {}


def test_report_lines_include_reference_and_variants(tmp_path):
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(REFERENCE), encoding="utf8")
    meas = m.measure_all(tmp_path, ["missing"], reference=ref)           # a missing variant is skipped
    assert list(meas) == ["_train_reference"]
    (tmp_path / "v").mkdir()
    meas["k_sharp"] = m.measure_variant(_variant(tmp_path / "v"))
    text = "\n".join(m.md_lines(meas))
    assert "TRAIN real" in text and "k_sharp" in text and "hf_ratio" in text and "% 0.6 - 1.5" in text
    assert meas["_train_reference"] == REFERENCE
