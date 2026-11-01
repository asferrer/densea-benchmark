"""INT8 calibration sets drawn from the TRAIN images of the group partition (never VAL or TEST).

Greedy selection balanced by class and by image source: at each step the image with the highest score
``sum(1 / (n_class + 1)) + 1 / (n_source + 1)`` is taken, where ``n_class`` and ``n_source`` count what has been
selected so far; ties are broken by a seeded shuffle of the pool (images sorted by id before shuffling). The sets
are nested: for ``--sizes 100 500`` the 100-image set is the first part of the 500-image set.

The script checks that no group and no file name of the selection appears in VAL or TEST. It stores the lists
(image id, group, source, classes), not the images; ``--copy-to`` copies the images of the largest set to a folder.
The list used in the paper is ``calibration/calib_100.json``.

Usage:
    python deploy_jetson/make_calibration_set.py [--sizes 100] [--seed 21] [--copy-to DIR]
Output: deploy_jetson/calibration/calib_<n>.json
"""
import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from common import paths  # noqa: E402

OUT = Path(__file__).resolve().parent / "calibration"


def load(split):
    with open(paths.annotations_path(split), encoding="utf8") as f:
        return json.load(f)


def index(train):
    """Images of a COCO file sorted by id and the sorted class names present in each image."""
    names = {c["id"]: c["name"] for c in train["categories"]}
    cats = collections.defaultdict(set)
    for a in train["annotations"]:
        cats[a["image_id"]].add(names[a["category_id"]])
    return sorted(train["images"], key=lambda im: im["id"]), {i: sorted(c) for i, c in cats.items()}


def greedy_order(imgs, cats, n, seed):
    """First ``n`` images of the class- and source-balanced greedy order."""
    rng = random.Random(seed)
    pool = imgs[:]
    rng.shuffle(pool)
    n_cat, n_src, order = collections.Counter(), collections.Counter(), []

    def score(im):
        return sum(1 / (n_cat[c] + 1) for c in cats.get(im["id"], [])) + 1 / (n_src[im["source"]] + 1)

    while pool and len(order) < n:
        im = pool.pop(max(range(len(pool)), key=lambda i: score(pool[i])))
        order.append(im)
        n_cat.update(cats.get(im["id"], []))
        n_src[im["source"]] += 1
    return order


def check_disjoint(order, held_out):
    """Exit if a selected image shares its group or its file name with a VAL or TEST image."""
    groups = {im["group"] for im in held_out}
    names = {Path(im["file_name"]).name.lower() for im in held_out}
    bad_g = [im["file_name"] for im in order if im["group"] in groups]
    bad_n = [im["file_name"] for im in order if Path(im["file_name"]).name.lower() in names]
    if bad_g or bad_n:
        sys.exit(f"overlap with VAL/TEST: {len(bad_g)} by group, {len(bad_n)} by file name "
                 f"(e.g. {(bad_g + bad_n)[:3]})")


def calibration_rows(selection, cats):
    return [{"id": im["id"], "group": im["group"], "source": im["source"], "classes": cats.get(im["id"], [])}
            for im in selection]


def write_sets(order, sizes, cats, seed):
    OUT.mkdir(exist_ok=True)
    for n in sizes:
        rows = calibration_rows(order[:n], cats)
        meta = {"split": "train", "seed": seed, "n": len(rows), "nested": True, "images": rows}
        with open(OUT / f"calib_{n}.json", "w", encoding="utf8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        by_class = collections.Counter(c for r in rows for c in r["classes"])
        by_source = collections.Counter(r["source"] for r in rows)
        print(f"calib_{n}: {len(rows)} images, {len({r['group'] for r in rows})} groups; "
              f"images per class {dict(by_class)}; per source {dict(by_source)}")


def copy_images(order, n, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for im in order[:n]:
        src = paths.DATA_ROOT / im["file_name"]
        shutil.copy2(src, dest / f"{im['id']:06d}_{src.name}")   # id prefix: avoids name clashes between sources
    print(f"copied {n} images to {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100])
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--copy-to", default=None, help="copy the images of the largest set to this folder")
    a = ap.parse_args()
    imgs, cats = index(load("train"))
    sizes = sorted(a.sizes)
    order = greedy_order(imgs, cats, sizes[-1], a.seed)
    check_disjoint(order, load("val")["images"] + load("test")["images"])
    write_sets(order, sizes, cats, a.seed)
    if a.copy_to:
        copy_images(order, sizes[-1], a.copy_to)


if __name__ == "__main__":
    main()
