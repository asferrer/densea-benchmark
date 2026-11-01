"""Replication on RF-DETR and DiffusionDet with 100% real data: AP on VAL and paired differences with 95% CIs from a
GROUP bootstrap over the group partition, with the same resamples as full_val_bootstrap.py. VAL only, never TEST.

Arms (seed 21): Real-Only (exp07), Minority-Focus with the clean-background batch (exp08) and Standard with the
copy-paste batch (exp06v11).
Predictions (CPU inference of the best checkpoint) are read from `predictions_dir(det, "val") / <exp>_val_preds.json`.
Usage: python analysis/full_val/replica_bootstrap.py --det rfdetr|diffusiondet [--preds DIR] [--b 2000]
Outputs: analysis/full_val/out/replica_<det>.md and replica_<det>.json
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

ARMS = {"exp07_real100": "Real-Only", "exp08_real_synth_minor": "MF clean-bg",
        "exp06v11_synth100_real100": "Std copy-paste"}
PAIRS = [("exp08_real_synth_minor", "exp07_real100"), ("exp06v11_synth100_real100", "exp07_real100"),
         ("exp08_real_synth_minor", "exp06v11_synth100_real100")]
OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--det", required=True, choices=["rfdetr", "diffusiondet"])
    ap.add_argument("--preds", default=None, help="folder with the predictions (default: predictions_dir)")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    preds = a.preds or paths.predictions_dir(a.det, "val")
    gt = load_gt(paths.annotations_path("val"))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    exps = [e for e in ARMS if (Path(preds) / f"{e}_val_preds.json").exists()]
    res = {e: evaluate(e, gt, preds, w) for e in exps}
    out = {"det": a.det, "n_img": len(ids), "n_groups": n_groups, "B": a.b, "runs": {}}
    md = [f"# Replication on {a.det}, 100% real: AP on VAL (n = {len(ids)} images, {n_groups} groups of the group "
          f"partition; group bootstrap, B = {a.b})", "", "| Run | Arm | AP | AP50 | AP75 | 95% CI AP |",
          "|---|---|---|---|---|---|"]
    for e in exps:
        o, (lo, hi) = res[e]["official"], 100 * np.percentile(res[e]["APb"], [2.5, 97.5])
        out["runs"][e] = {k: round(100 * v, 2) for k, v in o.items() if not isinstance(v, dict)}
        md.append(f"| {e} | {ARMS[e]} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75"))
                  + f" | [{lo:.2f}; {hi:.2f}] |")
    md += ["", "## Paired differences (same group resamples)", ""] + DIFF_HEAD
    for x, y in [q for q in PAIRS if q[0] in res and q[1] in res]:
        for m, _ in METRICS:
            md.append(diff_row(f"{ARMS[x]} − {ARMS[y]}", 100 * (res[x][m + "0"] - res[y][m + "0"]),
                               100 * (res[x][m + "b"] - res[y][m + "b"]), m, out, "pairs"))
    md += ["", "One seed (21) per arm: the CI reflects the variation of VAL, not of the seed."]
    OUT.mkdir(exist_ok=True)
    (OUT / f"replica_{a.det}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / f"replica_{a.det}.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
