"""Shared paths and constants of the benchmark kit.

Every location is configurable through an environment variable and defaults to a folder inside the
repository, so nothing depends on a particular machine.

Environment variables
---------------------
DATA_ROOT        Root of the DenSea-Aggregated package (the folder that contains ``annotations/`` and
                 ``images/``). Synthetic batches live in ``DATA_ROOT/synthetic/<batch>/``. Containers mount it
                 read-only as ``/data``.
ANNOTATIONS_DIR  Folder with ``train.json``, ``val.json`` and ``test.json`` (default ``DATA_ROOT/annotations``).
RUNS_DIR         Training outputs (weights, logs, predictions). Containers mount it as ``/runs``.
FEATURES_DIR     Image descriptors used to build the group partition (default ``data/features``).
PREPARE_OUT      Training sets written by ``prepare/prepare_strategies.py`` (default ``prepare/out``).
"""
import os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("DATA_ROOT", KIT / "data" / "densea"))
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", DATA_ROOT / "annotations"))
SYNTH_DIR = DATA_ROOT / "synthetic"
RUNS_DIR = Path(os.environ.get("RUNS_DIR", KIT / "runs"))
FEATURES_DIR = Path(os.environ.get("FEATURES_DIR", KIT / "data" / "features"))
PREPARE_OUT = Path(os.environ.get("PREPARE_OUT", KIT / "prepare" / "out"))

SPLIT_DIR = KIT / "split_group"
SPLIT_OUT = SPLIT_DIR / "out"          # split_ids.json, groups.json, REPORT.md (versioned)

CLASSES = ["Bottle", "Can", "Fishing_Net", "Glove", "Mask", "Metal_Debris", "Plastic_Debris", "Tire"]
SPLITS = ["train", "val", "test"]
SEED = 21


def annotations_path(split):
    """COCO file of a partition of DenSea-Aggregated."""
    return ANNOTATIONS_DIR / f"{split}.json"


def predictions_dir(detector, split):
    """Folder with ``<experiment>_<split>_preds.json`` files (COCO detections of the best checkpoint)."""
    return RUNS_DIR / "predictions" / detector / split
