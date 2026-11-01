"""Tests of the annotation pooling and of make_split_group on toy data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_group import make_split_group as msg  # noqa: E402
from split_group.pooling import pool_annotations  # noqa: E402

SOURCES = ["CleanSea", "Neural_Ocean", "Ocean_garbage"]


def _image(i, source, group):
    name = {"CleanSea": f"images/CleanSea/JPEGImages/HPD{1000 + i:04d}OUT0001.jpg",
            "Neural_Ocean": f"images/Neural_Ocean/train/n{i}_jpg.rf.{i:032x}.jpg",
            "Ocean_garbage": f"images/Ocean_garbage/train/o{i}_jpg.rf.{i:032x}.jpg"}[source]
    return {"id": i, "file_name": name, "source": source, "group": group}


def _write_toy(tmp_path, n=90):
    """n images in three files; ids are interleaved between files, the group field is arbitrary."""
    files = {s: {"images": [], "annotations": [], "categories": [{"id": c, "name": str(c)} for c in range(8)]}
             for s in ("train", "val", "test")}
    for i in range(n):
        split = ("train", "val", "test")[i % 3]
        files[split]["images"].append(_image(i, SOURCES[i % 3], n - i))
        files[split]["annotations"].append({"id": i, "image_id": i, "category_id": i % 8, "bbox": [0, 0, 1, 1]})
    for s, d in files.items():
        (tmp_path / f"{s}.json").write_text(json.dumps(d), encoding="utf8")
    return n


def test_pool_annotations_orders_by_id_and_keeps_fields(tmp_path):
    n = _write_toy(tmp_path)
    rows, classes = pool_annotations(tmp_path)
    assert [r["id"] for r in rows] == list(range(n))
    assert rows[4]["file_split"] == "val" and rows[4]["source"] == "Neural_Ocean" and rows[4]["group"] == n - 4
    assert classes[5] == [5]


def test_pool_annotations_rejects_duplicated_id(tmp_path):
    _write_toy(tmp_path, 6)
    d = json.loads((tmp_path / "val.json").read_text(encoding="utf8"))
    d["images"].append(_image(0, "CleanSea", 0))
    (tmp_path / "val.json").write_text(json.dumps(d), encoding="utf8")
    with pytest.raises(ValueError):
        pool_annotations(tmp_path)


def test_row_permutation_by_group(tmp_path):
    _write_toy(tmp_path, 9)
    rows, _ = pool_annotations(tmp_path)
    assert list(msg.row_permutation(rows, "id")) == list(range(9))
    perm = msg.row_permutation(rows, "group")
    assert [rows[i]["group"] for i in perm] == sorted(r["group"] for r in rows)
    rows[0]["group"] = None
    with pytest.raises(SystemExit):
        msg.row_permutation(rows, "group")


def _toy_features(n, seed=0):
    rng = np.random.default_rng(seed)
    hd = rng.integers(0, 256, (n, 8, 32), dtype=np.uint8)
    em = rng.normal(size=(n, 2048)).astype(np.float32)
    em /= np.linalg.norm(em, axis=1, keepdims=True)
    return hd, em.astype(np.float16)


def test_make_split_group_end_to_end(tmp_path, monkeypatch):
    ann, feat, out = tmp_path / "ann", tmp_path / "feat", tmp_path / "out"
    ann.mkdir()
    feat.mkdir()
    n = _write_toy(ann, 120)
    hd, em = _toy_features(n)
    em[1] = em[0]                                   # images 0 (CleanSea) and 1 (Neural_Ocean) look alike
    np.save(feat / "dhash.npy", hd)
    np.save(feat / "embeddings.npy", em)
    monkeypatch.setattr(sys, "argv", ["make_split_group.py", "--annotations-dir", str(ann),
                                      "--features-dir", str(feat), "--out", str(out), "--restarts", "3",
                                      "--order", "id"])
    msg.main()
    ids = json.loads((out / "split_ids.json").read_text())
    assert sorted(ids["train"] + ids["val"] + ids["test"]) == list(range(n))
    groups = json.loads((out / "groups.json").read_text())
    of = {i: g["group"] for g in groups for i in g["image_ids"]}
    assert of[0] == of[1]                           # linked by descriptor similarity
    split_of = {i: s for s in ("train", "val", "test") for i in ids[s]}
    assert all(split_of[i] == g["split"] for g in groups for i in g["image_ids"])
    assert (out / "REPORT.md").read_text(encoding="utf8").startswith("# Group partition report")


def test_make_split_group_needs_descriptors(tmp_path, monkeypatch):
    ann = tmp_path / "ann"
    ann.mkdir()
    _write_toy(ann, 9)
    monkeypatch.setattr(sys, "argv", ["make_split_group.py", "--annotations-dir", str(ann),
                                      "--features-dir", str(tmp_path / "none"), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        msg.main()
