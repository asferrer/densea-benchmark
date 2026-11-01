"""The three arms with 100% real data and 3 training seeds (21, 22, 23) per detector: AP, seed mean per arm and
paired differences between seed means with 95% CIs from a GROUP bootstrap over the group partition (same resamples
as full_val_bootstrap.py and replica_bootstrap.py). VAL by default. TEST is evaluated once, with the frozen
configuration: it needs --split test and ALLOW_TEST=1 (otherwise the script refuses to read TEST).

On VAL it also selects the model reported for deployment: the arm with the best mean AP over the 3 seeds and,
within it, the seed whose AP is closest to the mean.
Predictions (CPU inference of the best checkpoint) are read from
`predictions_dir(det, split) / <exp>[_s2|_s3]_<split>_preds.json`.
Usage: python analysis/full_val/seeds3_bootstrap.py --det rtdetrv2|rfdetr|diffusiondet [--preds DIR] [--b 2000]
       [--split val|test]
Outputs: analysis/full_val/out/seeds3_<det>.md and seeds3_<det>.json (on TEST: seeds3_<det>_test.*)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from eval.coco_tools import DETECTORS, load_gt  # noqa: E402
from full_val_bootstrap import diff_row  # noqa: E402
from px_val_bootstrap import METRICS, evaluate, group_weights  # noqa: E402

ARMS = {"exp07_real100": "Real-Only", "exp08_real_synth_minor": "MF clean-bg",
        "exp06v11_synth100_real100": "Std copy-paste"}
SEEDS = {"": 21, "_s2": 22, "_s3": 23}
PAIRS = [("exp08_real_synth_minor", "exp07_real100"), ("exp06v11_synth100_real100", "exp07_real100"),
         ("exp08_real_synth_minor", "exp06v11_synth100_real100")]
OUT = Path(__file__).resolve().parent / "out"


def arm_rows(res, arms, out):
    """Seed mean per arm (CI of the group bootstrap over the mean), SD across seeds and the seed closest to the mean."""
    md = ["", "## Mean of 3 seeds per arm", "",
          "| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |",
          "|---|---|---|---|---|---|---|"]
    for arm, runs in arms.items():
        aps = np.array([100 * res[e]["AP0"] for e in runs])
        mb = 100 * np.mean([res[e]["APb"] for e in runs], axis=0)
        lo, hi = np.percentile(mb, [2.5, 97.5])
        near = runs[int(np.argmin(np.abs(aps - aps.mean())))]
        m50, m75 = (100 * np.mean([res[e][m + "0"] for e in runs]) for m in ("AP50", "AP75"))
        out["arms"][arm] = {"runs": runs, "AP": round(aps.mean(), 2), "ci": [round(lo, 2), round(hi, 2)],
                            "sd_seeds": round(aps.std(ddof=1), 2), "AP50": round(m50, 2), "AP75": round(m75, 2),
                            "nearest": near}
        md.append(f"| {ARMS[arm]} | {aps.mean():.2f} | [{lo:.2f}; {hi:.2f}] | {aps.std(ddof=1):.2f} | {m50:.2f} | "
                  f"{m75:.2f} | {near} ({100 * res[near]['AP0']:.2f}) |")
    return md


def pair_rows(res, arms, out):
    md = ["", "## Differences between seed means (same group resamples)", "",
          "| A − B | Metric | Difference | 95% CI | p (two-sided) |", "|---|---|---|---|---|"]
    for x, y in [q for q in PAIRS if q[0] in arms and q[1] in arms]:
        for m, _ in METRICS:
            d0 = 100 * (np.mean([res[e][m + "0"] for e in arms[x]]) - np.mean([res[e][m + "0"] for e in arms[y]]))
            db = 100 * (np.mean([res[e][m + "b"] for e in arms[x]], axis=0)
                        - np.mean([res[e][m + "b"] for e in arms[y]], axis=0))
            md.append(diff_row(f"{ARMS[x]} − {ARMS[y]}", d0, db, m, out, "pairs"))
    return md


def seed_rows(res, out):
    """Paired differences between arms within each seed (AP only)."""
    md = ["", "## Differences within each seed (AP)", "",
          "| A − B | Seed | Difference | 95% CI | p (two-sided) |", "|---|---|---|---|---|"]
    for s, seed in SEEDS.items():
        for x, y in [(x + s, y + s) for x, y in PAIRS if x + s in res and y + s in res]:
            d0, db = 100 * (res[x]["AP0"] - res[y]["AP0"]), 100 * (res[x]["APb"] - res[y]["APb"])
            row = diff_row(f"{ARMS[x.removesuffix(s) if s else x]} − {ARMS[y.removesuffix(s) if s else y]}", d0, db,
                           "AP", out, f"seed{seed}")
            md.append(row.replace(" | AP | ", f" | {seed} | "))
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--det", required=True, choices=list(DETECTORS))
    ap.add_argument("--preds", default=None, help="folder with the predictions (default: predictions_dir)")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    if a.split == "test" and os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    preds = a.preds or paths.predictions_dir(a.det, a.split)
    gt = load_gt(paths.annotations_path(a.split))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    runs = [e + s for e in ARMS for s in SEEDS if (Path(preds) / f"{e}{s}_{a.split}_preds.json").exists()]
    res = {e: evaluate(e, gt, preds, w, a.split) for e in runs}
    SPLIT, TAG = a.split.upper(), "" if a.split == "val" else "_test"
    arms = {e: [e + s for s in SEEDS] for e in ARMS if all(e + s in res for s in SEEDS)}
    out = {"det": a.det, "n_img": len(ids), "n_groups": n_groups, "B": a.b, "runs": {}, "arms": {}}
    md = [f"# {a.det}, 100% real, 3 seeds: AP on {SPLIT} (n = {len(ids)} images, {n_groups} groups of the group "
          f"partition; group bootstrap, B = {a.b})", "", "| Run | Seed | AP | AP50 | AP75 | 95% CI AP |",
          "|---|---|---|---|---|---|"]
    for e in runs:
        o, (lo, hi) = res[e]["official"], 100 * np.percentile(res[e]["APb"], [2.5, 97.5])
        out["runs"][e] = {k: round(100 * v, 2) for k, v in o.items() if not isinstance(v, dict)}
        seed = SEEDS.get(e[-3:], 21)
        md.append(f"| {e} | {seed} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75"))
                  + f" | [{lo:.2f}; {hi:.2f}] |")
    md += arm_rows(res, arms, out) + pair_rows(res, arms, out) + seed_rows(res, out)
    if arms and a.split == "val":
        best = max(out["arms"], key=lambda k: out["arms"][k]["AP"])
        out["deployment"] = {"arm": best, "run": out["arms"][best]["nearest"]}
        md += ["", f"Model selected for deployment (best mean AP; seed closest to the mean): "
                   f"{ARMS[best]}, {out['arms'][best]['nearest']}."]
    md += ["", f"The CI of the mean reflects the variation of {SPLIT} with the 3 seeds used; the SD across seeds "
               "reflects the variation of training."]
    OUT.mkdir(exist_ok=True)
    (OUT / f"seeds3_{a.det}{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / f"seeds3_{a.det}{TAG}.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
