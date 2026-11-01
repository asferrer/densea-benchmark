"""Tests of verify_groups on toy partitions."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_group import verify_groups as vg  # noqa: E402

ARGS = argparse.Namespace(td=24, tc=0.95, tc_rf=0.86, device="cpu")


def _row(i, name, split, source="Ocean_garbage", group=None):
    return {"id": i, "file_name": name, "source": source, "file_split": split, "group": group}


def _clean_rows():
    return [_row(0, "a_jpg.rf." + "0" * 32 + ".jpg", "train", group=0),
            _row(1, "b_jpg.rf." + "1" * 32 + ".jpg", "val", group=1),
            _row(2, "c_jpg.rf." + "2" * 32 + ".jpg", "test", group=2),
            _row(3, "JPEGImages/HPD0001OUT0001.jpg", "train", "CleanSea", 3),
            _row(4, "JPEGImages/HPD0002OUT0001.jpg", "val", "CleanSea", 4)]


def test_clean_partition_passes():
    res = vg.run_checks(_clean_rows(), ARGS)
    assert res["violations"] == 0
    assert res["images"] == {"train": 2, "val": 2, "test": 1}


def test_same_name_key_across_partitions_is_detected():
    rows = _clean_rows()
    rows.append(_row(5, "a0_a_jpg.rf." + "f" * 32 + ".jpg", "test", group=5))     # same photograph as row 0
    res = vg.run_checks(rows, ARGS)
    assert res["name_keys"]["train-test"] == {"keys": 1, "images": 2}
    assert res["violations"] >= 1


def test_cleansea_dive_across_partitions_is_detected():
    rows = _clean_rows()
    rows.append(_row(5, "JPEGImages/HPD0001HDTV0009.jpg", "test", "CleanSea", 6))   # same dive as row 3
    res = vg.run_checks(rows, ARGS)
    assert res["cleansea_dive"]["train-test"]["keys"] == 1
    assert res["cleansea_dive_camera"]["train-test"]["keys"] == 0


def test_shared_group_id_is_detected():
    rows = _clean_rows()
    rows[1]["group"] = 0
    res = vg.run_checks(rows, ARGS)
    assert res["group_ids"]["train-val"]["keys"] == 1


def test_group_check_skipped_without_group_field():
    rows = _clean_rows()
    for r in rows:
        r["group"] = None
    assert vg.run_checks(rows, ARGS)["group_ids"] is None


def _descriptors(n):
    rng = np.random.default_rng(1)
    hd = np.unpackbits(rng.integers(0, 256, (n, 8, 32), dtype=np.uint8), axis=2)
    em = rng.normal(size=(n, 64)).astype(np.float32)
    return hd, em / np.linalg.norm(em, axis=1, keepdims=True)


def test_similar_images_across_partitions_are_detected():
    rows = _clean_rows()
    hd, em = _descriptors(len(rows))
    clean = vg.run_checks(rows, ARGS, hd, em)
    assert clean["violations"] == 0
    em[2] = em[0]                                    # test image identical in content to a train image
    res = vg.run_checks(rows, ARGS, hd, em)
    assert res["content"]["partition criterion"]["train-test"]["pairs"] == 1
    assert res["violations"] == 1


def test_markdown_report_lists_checks(tmp_path):
    res = vg.run_checks(_clean_rows(), ARGS, *_descriptors(5))
    vg.write_md(res, tmp_path)
    text = (tmp_path / "verification.md").read_text(encoding="utf8")
    assert "Violations: **0**" in text and "partition criterion" in text
