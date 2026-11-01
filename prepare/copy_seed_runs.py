"""Copy the training files of an experiment to the experiments that reuse them with another training seed.

Experiments with a `data_from` field (prepare/experiments.py) get a copy of PREPARE_OUT/<data_from> in
PREPARE_OUT/<exp>: images and annotations are untouched and only info.description records the origin. The
training seed is read from `train_seed` by the launchers. An existing folder is never overwritten.

Usage: python prepare/copy_seed_runs.py [--phase 8]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402

OUT = paths.PREPARE_OUT


def copy_run(name, src):
    dst = OUT / name
    if dst.exists():
        print(name, "already exists; left untouched")
        return
    dst.mkdir()
    for split in paths.SPLITS:
        d = json.load(open(OUT / src / f"{split}.json", encoding="utf8"))
        info = d.setdefault("info", {})
        info["description"] = (f"{info.get('description', src)} | copy for {name} (training seed "
                               f"{EXPERIMENTS[name]['train_seed']}, same data as {src})")
        json.dump(d, open(dst / f"{split}.json", "w", encoding="utf8"), ensure_ascii=False)
        print(name, split, len(d["images"]), "images", len(d["annotations"]), "annotations")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", type=int, default=8)
    a = ap.parse_args()
    for name, exp in EXPERIMENTS.items():
        if exp["phase"] == a.phase and "data_from" in exp:
            copy_run(name, exp["data_from"])


if __name__ == "__main__":
    main()
