"""DiffusionDet sampling steps: AP on VAL with 1, 2 and 4 sampling steps of the 9 final models (GPU inference), mean
of the 3 seeds per arm and paired difference "k steps - 1 step" of the mean, with 95% CIs from a GROUP bootstrap over
the group partition (same resamples as seeds3_bootstrap.py). VAL only.

Predictions: `predictions_dir("diffusiondet", "val") / steps<k> / <exp>[_s2|_s3]_val_preds.json`.
Usage: python analysis/full_val/dd_steps_bootstrap.py [--preds DIR] [--b 2000]
Outputs: analysis/full_val/out/dd_steps.md and dd_steps.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from eval.coco_tools import load_gt  # noqa: E402
from full_val_bootstrap import diff_row  # noqa: E402
from px_val_bootstrap import DIFF_HEAD, METRICS, evaluate, group_weights  # noqa: E402
from seeds3_bootstrap import ARMS, OUT, SEEDS  # noqa: E402

STEPS = (1, 2, 4)


def unit(k):
    return f"{k} step" + ("s" if k > 1 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=str(paths.predictions_dir("diffusiondet", "val")),
                    help="folder with the steps<k>/ subfolders")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    gt = load_gt(paths.annotations_path("val"))
    w, n_groups = group_weights(sorted(gt.getImgIds()), a.b)
    res = {(k, e + s): evaluate(e + s, gt, Path(a.preds) / f"steps{k}", w) for k in STEPS for e in ARMS for s in SEEDS}
    out = {"n_groups": n_groups, "B": a.b, "runs": {}, "arms": {}, "pairs": []}
    md = [f"# DiffusionDet, 100% real, 3 seeds: AP on VAL with 1, 2 and 4 sampling steps (GPU; {n_groups} groups "
          f"of the group partition; group bootstrap, B = {a.b})", "",
          "| Run | " + " | ".join(f"AP {unit(k)}" for k in STEPS) + " |", "|---|" + "---|" * len(STEPS)]
    for e in ARMS:
        for s in SEEDS:
            out["runs"][e + s] = {k: round(100 * res[(k, e + s)]["AP0"], 2) for k in STEPS}
            md.append(f"| {e + s} | " + " | ".join(f"{100 * res[(k, e + s)]['AP0']:.2f}" for k in STEPS) + " |")
    md += ["", "## Mean of 3 seeds per arm", "", "| Arm | " + " | ".join(
        f"AP {unit(k)} | AP50 {unit(k)} | AP75 {unit(k)}" for k in STEPS) + " |", "|---|" + "---|" * 3 * len(STEPS)]
    for e, arm in ARMS.items():
        cells, o = [], {}
        for k in STEPS:
            for m, _ in METRICS:
                v = 100 * np.mean([res[(k, e + s)][m + "0"] for s in SEEDS])
                cells.append(f"{v:.2f}"); o[f"{m}_{k}"] = round(v, 2)
        out["arms"][arm] = o
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    md += ["", "## k steps − 1 step (seed mean; same group resamples)", ""] + DIFF_HEAD
    for e, arm in ARMS.items():
        for k in STEPS[1:]:
            for m, _ in METRICS:
                d0 = 100 * np.mean([res[(k, e + s)][m + "0"] - res[(1, e + s)][m + "0"] for s in SEEDS])
                db = 100 * np.mean([res[(k, e + s)][m + "b"] - res[(1, e + s)][m + "b"] for s in SEEDS], axis=0)
                md.append(diff_row(f"{arm}: {unit(k)} − 1 step", d0, db, m, out, "pairs"))
    OUT.mkdir(exist_ok=True)
    (OUT / "dd_steps.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / "dd_steps.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
