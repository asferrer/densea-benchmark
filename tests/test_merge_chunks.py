"""Tests of synthetic/merge_chunks.py on toy chunks."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic import merge_chunks as mc  # noqa: E402

CATS = [{"id": 0, "name": "Can"}, {"id": 1, "name": "Tire"}]


def _make_chunk(src, k, n_img, variant="s_drop"):
    d = src / f"chunk_{k}" / variant
    (d / "images").mkdir(parents=True)
    images = [{"id": i, "file_name": f"synthetic_{i:05d}.jpg"} for i in range(n_img)]
    anns = [{"id": i, "image_id": i, "category_id": i % 2, "bbox": [0, 0, 2, 2],
             "origin": "pasted" if i % 2 else "real"} for i in range(n_img)]
    for im in images:
        (d / "images" / im["file_name"]).write_bytes(b"x")
    (d / "synthetic_dataset.json").write_text(json.dumps({"categories": CATS, "images": images, "annotations": anns}))
    (d / "variant.json").write_text(json.dumps({"rejected": k, "seconds": 3600 * k}))


def _setup(tmp_path):
    src = tmp_path / "src"
    _make_chunk(src, 1, 3)
    _make_chunk(src, 2, 2)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"targets_per_class": {"Can": 5, "Tire": 5}}))
    return src, request


def test_merge_renumbers_ids_and_rewrites_paths(tmp_path):
    src, _ = _setup(tmp_path)
    chunks = [(f"chunk_{k}", mc.load_chunk(src, f"chunk_{k}", "s_drop")[0]) for k in (1, 2)]
    ds = mc.merge(chunks, "toy")
    assert [im["id"] for im in ds["images"]] == list(range(5))
    assert [a["id"] for a in ds["annotations"]] == list(range(5))
    assert ds["images"][3]["file_name"] == "chunk_2/images/synthetic_00000.jpg" and ds["images"][3]["chunk"] == "chunk_2"
    assert ds["annotations"][3]["image_id"] == 3


def test_main_copies_images_and_writes_summary(tmp_path, monkeypatch):
    src, request = _setup(tmp_path)
    dst = tmp_path / "dst"
    monkeypatch.setattr(sys, "argv", ["merge_chunks.py", "--src", str(src), "--dst", str(dst), "--chunks", "2",
                                      "--request", str(request), "--name", "toy"])
    mc.main()
    assert len(list(dst.glob("chunk_*/images/*.jpg"))) == 5
    summary = json.loads((dst / "summary.json").read_text())
    assert summary["images"] == 5 and summary["annotations"] == 5
    assert summary["pasted_total"] == 2 and summary["real_total"] == 3
    assert summary["target_per_class"] == {"Can": 5, "Tire": 5}
    assert summary["rejected_per_chunk"] == {"chunk_1": 1, "chunk_2": 2}
    assert summary["hours_per_chunk"] == {"chunk_1": 1.0, "chunk_2": 2.0}
    assert json.loads((dst / "synthetic_dataset.json").read_text())["images"][0]["file_name"] == \
        "chunk_1/images/synthetic_00000.jpg"


def test_existing_destination_is_not_overwritten(tmp_path, monkeypatch):
    src, request = _setup(tmp_path)
    dst = tmp_path / "dst"
    dst.mkdir()
    monkeypatch.setattr(sys, "argv", ["merge_chunks.py", "--src", str(src), "--dst", str(dst), "--chunks", "2",
                                      "--request", str(request), "--name", "toy"])
    with pytest.raises(SystemExit):
        mc.main()
    assert not any(dst.iterdir())


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    src, request = _setup(tmp_path)
    dst = tmp_path / "dst"
    monkeypatch.setattr(sys, "argv", ["merge_chunks.py", "--src", str(src), "--dst", str(dst), "--chunks", "2",
                                      "--request", str(request), "--name", "toy", "--dry-run"])
    mc.main()
    assert not dst.exists()
