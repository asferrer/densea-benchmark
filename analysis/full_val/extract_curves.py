"""VAL AP per epoch of RT-DETRv2 trained on 100% real data, for the learning-curve figures.

Reads `RUNS_DIR/rtdetrv2/<exp>/log.txt` (one JSON line per epoch; `test_coco_eval_bbox` is the EMA AP on VAL of the
group partition, the one that selects best.pth) and the size of `PREPARE_OUT/<exp>/train.json`, and writes
out/curves_full.csv with the epoch, the cumulative processed images (epoch x training images, in millions) and
AP/AP50. VAL only.
Usage: python analysis/full_val/extract_curves.py [--runs DIR]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
from common import paths  # noqa: E402

EXPS = ["exp07_real100", "exp06_synth100_real100", "exp08_real_synth_minor", "exp09_real_synth_invprop",
        "exp06v11_synth100_real100", "exp08v11_real_synth_minor", "exp09v11_real_synth_invprop",
        "exp07_real100_s2", "exp08_real_synth_minor_s2", "exp06v11_synth100_real100_s2"]
OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(paths.RUNS_DIR / "rtdetrv2"), help="folder with the <exp>/log.txt files")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for e in EXPS:
        log = Path(a.runs) / e / "log.txt"
        if not log.exists():
            continue
        n = len(json.load(open(paths.PREPARE_OUT / e / "train.json", encoding="utf8"))["images"])
        for line in open(log, encoding="utf8"):
            d = json.loads(line)
            ep = d["epoch"] + 1  # log.txt counts from 0; here epoch 1 is the first
            rows.append({"exp": e, "epoch": ep, "n_train": n, "images_seen_M": round(ep * n / 1e6, 4),
                         "AP": round(100 * d["test_coco_eval_bbox"][0], 3),
                         "AP50": round(100 * d["test_coco_eval_bbox"][1], 3)})
        best = max((r for r in rows if r["exp"] == e), key=lambda r: r["AP"])
        print(f"{e}: {n} images, epoch {best['epoch']} is the best (AP {best['AP']:.2f})")
    if not rows:
        sys.exit(f"no log.txt found under {a.runs}")
    with open(OUT / "curves_full.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
