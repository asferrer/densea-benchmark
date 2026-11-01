"""Tests of the training-set builders on toy data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from prepare import prepare_strategies as ps  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402

CATS = [{"id": c, "name": n, "supercategory": "debris"} for c, n in enumerate(paths.CLASSES)]


def _write_split(folder, split, ids):
    images = [{"id": i, "file_name": f"images/Ocean_garbage/train/{i}.jpg", "source": "Ocean_garbage",
               "group": i, "in_package": True} for i in ids]
    anns = [{"id": i, "image_id": i, "category_id": i % 8, "bbox": [0, 0, 1, 1]} for i in ids]
    (folder / f"{split}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": CATS,
                                                      "licenses": [{"id": 1}], "info": {}}))


def test_load_real_keeps_relative_image_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ANNOTATIONS_DIR", tmp_path)
    for split, ids in (("train", range(0, 8)), ("val", range(8, 10)), ("test", range(10, 12))):
        _write_split(tmp_path, split, ids)
    real = ps.load_real()
    assert set(real) == {"train", "val", "test"}
    assert real["train"]["images"][0]["file_name"] == "images/Ocean_garbage/train/0.jpg"
    assert real["train"]["licenses"] == [] and len(real["val"]["images"]) == 2


def test_load_synth_prefixes_batch_folder_and_drops_empty_images(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SYNTH_DIR", tmp_path)
    (tmp_path / "clean_background").mkdir()
    ds = {"categories": [{"id": 0, "name": "Can"}, {"id": 1, "name": "Tire"}],
          "images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"}],
          "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}]}
    (tmp_path / "clean_background" / "synthetic_dataset.json").write_text(json.dumps(ds))
    synth, dropped = ps.load_synth("v9", CATS)
    assert [im["file_name"] for im in synth["images"]] == ["synthetic/clean_background/images/a.jpg"]
    assert dropped == 1 and synth["annotations"][0]["category_id"] == paths.CLASSES.index("Tire")


def test_every_batch_label_of_the_experiments_has_a_folder():
    labels = {e.get("lot", "v9") for e in EXPERIMENTS.values()} | {e["match"] for e in EXPERIMENTS.values() if "match" in e}
    assert labels <= set(ps.BATCHES)


def test_experiment_registry_fields():
    allowed = {"real", "synth", "phase", "lot", "train_seed", "data_from", "match", "label"}
    for name, e in EXPERIMENTS.items():
        assert set(e) <= allowed, name
        assert "data_from" not in e or e["data_from"] in EXPERIMENTS
