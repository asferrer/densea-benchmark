"""Tests of external_seabed: class mapping and annotation conversions (no data, no images)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import external_seabed as ex  # noqa: E402


def test_mapping_targets_are_our_classes_and_unmapped_dropped():
    assert set(ex.CLASS_MAP.values()) <= set(ex.dg.CLASSES)
    assert ex.map_class("Bottle") == "Bottle" and ex.map_class("Tire") == "Tire"
    for dropped in ("Wood", "Squared_Can", "Metal_Chain", "Plastic_Bag", "Mask"):
        assert ex.map_class(dropped) is None


def test_ext_class_ids_match_zenodo_order():
    assert len(ex.EXT_CLASSES) == 19
    assert ex.EXT_CLASSES[3] == "Bottle" and ex.EXT_CLASSES[14] == "Metal_Debris" and ex.EXT_CLASSES[15] == "Tire"


def test_yolo_to_coco_and_polygon_area():
    i, box = ex.yolo_to_coco(["3", "0.5", "0.5", "0.25", "0.5"], 100, 100)
    assert i == 3 and box == [38, 25, 25, 50]
    assert abs(ex.poly_area_norm([0, 0, 1, 0, 1, 1, 0, 1]) - 1.0) < 1e-9
