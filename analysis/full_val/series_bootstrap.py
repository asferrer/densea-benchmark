"""Incremental series: AP on VAL of RT-DETRv2 per real-data fraction (10/25/50/100%) and paired differences with
95% CIs from a GROUP bootstrap over the group partition, with the same resamples as px_val_bootstrap.py and
full_val_bootstrap.py (same images and resampling seed). VAL only, never TEST.

At each fraction there are three arms: real only, clean-background batch (v9) and copy-paste batch (v11). When an arm
has several seeds (21 and 22; at 100%, 21, 22 and 23) the mean is used, with the same resample. Standard strategy at
every fraction:
  10%   px_real10 | px_v9_real10 (+_s2) | px_p_real10 (+_s2)      matched in pasted objects; copy-paste p_pos_tight
  25%   inc_real25 (+_s2) | inc_v9_real25 (+_s2) | inc_v11_real25 (+_s2)   matched; copy-paste s_drop
  50%   inc_real50 | inc_v9_real50 | inc_v11_real50              matched in pasted objects; copy-paste s_drop
  100%  exp07 (+_s2, _s3) | exp06 (seed 21 only) | exp06v11 (+_s2, _s3)   NOT matched (40,004 vs 21,238 pasted objects)
Separate reference at 100%: Minority-Focus with the clean-background batch (exp08 + _s2 + _s3).
Predictions (CPU inference, analysis/box_bias/infer_val_cpu.py) are read from
`predictions_dir("rtdetrv2", "val") / <exp>_val_preds.json`; runs without predictions are omitted.
Usage: python analysis/full_val/series_bootstrap.py [--preds DIR] [--b 2000]
Outputs: analysis/full_val/out/series.md, series.json and series.csv (for the figures)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
from common import paths  # noqa: E402
from eval.coco_tools import load_gt  # noqa: E402
from px_val_bootstrap import METRICS, evaluate, group_weights  # noqa: E402

SERIES = {
    10: {"real": ["px_real10"], "v9": ["px_v9_real10", "px_v9_real10_s2"], "cp": ["px_p_real10", "px_p_real10_s2"]},
    25: {"real": ["inc_real25", "inc_real25_s2"], "v9": ["inc_v9_real25", "inc_v9_real25_s2"],
         "cp": ["inc_v11_real25", "inc_v11_real25_s2"]},
    50: {"real": ["inc_real50"], "v9": ["inc_v9_real50"], "cp": ["inc_v11_real50"]},
    100: {"real": ["exp07_real100", "exp07_real100_s2", "exp07_real100_s3"], "v9": ["exp06_synth100_real100"],
          "cp": ["exp06v11_synth100_real100", "exp06v11_synth100_real100_s2", "exp06v11_synth100_real100_s3"],
          "v9_mf": ["exp08_real_synth_minor", "exp08_real_synth_minor_s2", "exp08_real_synth_minor_s3"]},
}
PAIRS = [("v9", "real"), ("cp", "real"), ("v9", "cp"), ("v9_mf", "real"), ("v9_mf", "cp")]
NAMES = {"real": "Real-Only", "v9": "clean-bg", "cp": "copy-paste", "v9_mf": "MF clean-bg"}
OUT = Path(__file__).resolve().parent / "out"


def ci(db):
    lo, hi = np.percentile(db, [2.5, 97.5])
    return round(lo, 2), round(hi, 2), round(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean())), 4)


def arm(res, exps, key):
    """Mean over the runs of the arm: official value (scalar) and resamples (vector)."""
    return (100 * np.mean([res[e][key + "0"] for e in exps]),
            100 * np.mean([res[e][key + "b"] for e in exps], axis=0))


def arm_table(res, out):
    """Markdown table with one row per (fraction, arm); fills `out["arms"]`."""
    md = ["| Real | Arm | Runs | AP | AP50 | AP75 | 95% CI AP |", "|---|---|---|---|---|---|---|"]
    for frac, arms in SERIES.items():
        for k, exps in arms.items():
            vals = {m: arm(res, exps, m) for m, _ in METRICS}
            lo, hi, _ = ci(vals["AP"][1])
            out["arms"][f"{frac}:{k}"] = {"exps": exps, **{m: round(v[0], 2) for m, v in vals.items()},
                                          "ci_AP": [lo, hi]}
            md.append(f"| {frac}% | {NAMES[k]} | {' + '.join(exps)} | " +
                      " | ".join(f"{vals[m][0]:.2f}" for m, _ in METRICS) + f" | [{lo:.2f}; {hi:.2f}] |")
    return md


def pair_table(res, out):
    """Markdown table of the paired differences per fraction; fills `out["pairs"]`; returns (lines, csv rows)."""
    md = ["| Real | A − B | Metric | Difference | 95% CI | p (two-sided) |", "|---|---|---|---|---|---|"]
    rows = []
    for frac, arms in SERIES.items():
        for x, y in [p for p in PAIRS if p[0] in arms and p[1] in arms]:
            for m, _ in METRICS:
                (x0, xb), (y0, yb) = arm(res, arms[x], m), arm(res, arms[y], m)
                lo, hi, p = ci(xb - yb)
                out["pairs"].append({"real": frac, "a": x, "b": y, "metric": m, "diff": round(x0 - y0, 2),
                                     "ci": [lo, hi], "p": p})
                md.append(f"| {frac}% | {NAMES[x]} − {NAMES[y]} | {m} | {x0 - y0:+.2f} | [{lo:+.2f}; {hi:+.2f}] | {p:.4f} |")
                rows.append({"real": frac, "a": x, "b": y, "metric": m, "diff": round(x0 - y0, 3),
                             "lo": lo, "hi": hi, "p": p})
    return md, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", default=str(paths.predictions_dir("rtdetrv2", "val")),
                    help="folder with <exp>_val_preds.json")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    gt = load_gt(paths.annotations_path("val"))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    for arms in SERIES.values():        # runs without predictions are omitted
        for k, exps in arms.items():
            arms[k] = [e for e in exps if (Path(a.preds) / f"{e}_val_preds.json").exists()]
    res = {}
    for arms in SERIES.values():
        for e in sum(arms.values(), []):
            res[e] = evaluate(e, gt, a.preds, w)
    out = {"n_img": len(ids), "n_groups": n_groups, "B": a.b, "arms": {}, "pairs": []}
    md = [f"# Incremental series: AP on VAL per real fraction (n = {len(ids)} images, {n_groups} groups; group "
          f"bootstrap, B = {a.b})", ""] + arm_table(res, out)
    pairs_md, rows = pair_table(res, out)
    md += ["", "## Paired differences per fraction (same group resamples)", ""] + pairs_md
    md += ["", "One seed per arm except where several runs are listed (mean of seeds 21 and 22; at 100%, 21, 22 and "
           "23). The CI reflects the variation of VAL, not of the seed (measured seed-to-seed noise: 1-1.5 AP). At "
           "100% the batches are not matched in pasted objects."]
    OUT.mkdir(exist_ok=True)
    (OUT / "series.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / "series.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
    with open(OUT / "series.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
