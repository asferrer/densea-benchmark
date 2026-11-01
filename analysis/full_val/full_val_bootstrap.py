"""RT-DETRv2 trained on 100% of the real data: AP on VAL and paired differences with 95% CIs from a GROUP bootstrap
over the group partition, with the same resamples as the transfer check (px_val_bootstrap.py). VAL only, never TEST.

Compares the strategies built with the clean-background batch (v9; exp06/08/09) and with the copy-paste batch
(v11; exp06v11/08v11/09v11) against Real-Only (exp07). With training seed 22 (suffix _s2) it adds the seed means.
Predictions: CPU inference of best.pth (analysis/box_bias/infer_val_cpu.py), read from
`predictions_dir("rtdetrv2", "val") / <exp>_val_preds.json`.
Usage: python analysis/full_val/full_val_bootstrap.py [--preds DIR] [--b 2000]
Outputs: analysis/full_val/out/full_val.md and full_val.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
from common import paths  # noqa: E402
from eval.coco_tools import load_gt  # noqa: E402
from px_val_bootstrap import DIFF_HEAD, METRICS, evaluate, group_weights  # noqa: E402

REAL = "exp07_real100"
V9 = ["exp06_synth100_real100", "exp08_real_synth_minor", "exp09_real_synth_invprop"]
V11 = ["exp06v11_synth100_real100", "exp08v11_real_synth_minor", "exp09v11_real_synth_invprop"]
S2 = [REAL + "_s2", "exp08_real_synth_minor_s2", "exp06v11_synth100_real100_s2"]
EXPS = [REAL] + V9 + V11 + S2
PAIRS = ([(e, REAL) for e in V9 + V11]
         + [(a, b) for a, b in zip(V9, V11)]                        # same strategy, other batch
         + [("exp08_real_synth_minor", "exp06v11_synth100_real100"),  # best of each batch (seed 21)
            (S2[1], S2[0]), (S2[2], S2[0]), (S2[1], S2[2])])
# Mean of seeds (21 and 22) per arm, with the same group resample.
SEED_MEANS = {"Real-Only (mean)": [REAL, S2[0]],
              "MF clean-bg (mean)": ["exp08_real_synth_minor", S2[1]],
              "Std copy-paste (mean)": ["exp06v11_synth100_real100", S2[2]]}
SEED_PAIRS = [("MF clean-bg (mean)", "Real-Only (mean)"), ("Std copy-paste (mean)", "Real-Only (mean)"),
              ("MF clean-bg (mean)", "Std copy-paste (mean)")]
OUT = Path(__file__).resolve().parent / "out"


def diff_row(name, d0, db, metric, out, key):
    """Markdown row of a paired difference (point estimate `d0`, resamples `db`) and its record in `out[key]`."""
    lo, hi = np.percentile(db, [2.5, 97.5])
    p = min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))
    out.setdefault(key, []).append({"pair": name, "metric": metric, "diff": round(d0, 2),
                                    "ci": [round(lo, 2), round(hi, 2)], "p": round(p, 4)})
    return f"| {name} | {metric} | {d0:+.2f} | [{lo:+.2f}; {hi:+.2f}] | {p:.4f} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=str(paths.predictions_dir("rtdetrv2", "val")),
                    help="folder with <exp>_val_preds.json")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    gt = load_gt(paths.annotations_path("val"))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    exps = [e for e in EXPS if (Path(a.preds) / f"{e}_val_preds.json").exists()]
    res = {e: evaluate(e, gt, a.preds, w) for e in exps}
    out = {"n_img": len(ids), "n_groups": n_groups, "B": a.b, "runs": {}}
    md = [f"# 100% real: AP on VAL (n = {len(ids)} images, {n_groups} groups of the group partition; "
          f"group bootstrap, B = {a.b})", "", "| Run | AP | AP50 | AP75 | 95% CI AP |", "|---|---|---|---|---|"]
    for e in exps:
        o, (lo, hi) = res[e]["official"], 100 * np.percentile(res[e]["APb"], [2.5, 97.5])
        out["runs"][e] = {k: round(100 * v, 2) for k, v in o.items() if not isinstance(v, dict)}
        md.append(f"| {e} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75")) + f" | [{lo:.2f}; {hi:.2f}] |")
    md += ["", "## Paired differences (same group resamples)", ""] + DIFF_HEAD
    for x, y in [q for q in PAIRS if q[0] in res and q[1] in res]:
        for m, _ in METRICS:
            md.append(diff_row(f"{x} − {y}", 100 * (res[x][m + "0"] - res[y][m + "0"]),
                               100 * (res[x][m + "b"] - res[y][m + "b"]), m, out, "pairs"))
    arms = {k: v for k, v in SEED_MEANS.items() if all(e in res for e in v)}
    if arms:
        md += ["", "## Seed means per arm", ""] + DIFF_HEAD
    for x, y in [q for q in SEED_PAIRS if q[0] in arms and q[1] in arms]:
        for m, _ in METRICS:
            d0 = 100 * (np.mean([res[e][m + "0"] for e in arms[x]]) - np.mean([res[e][m + "0"] for e in arms[y]]))
            db = 100 * (np.mean([res[e][m + "b"] for e in arms[x]], axis=0)
                        - np.mean([res[e][m + "b"] for e in arms[y]], axis=0))
            md.append(diff_row(f"{x} − {y}", d0, db, m, out, "seed_means"))
    md += ["", "The CI of each run reflects the variation of VAL, not of the seed. Runs without the _s2 suffix are seed 21."]
    OUT.mkdir(exist_ok=True)
    (OUT / "full_val.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / "full_val.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
