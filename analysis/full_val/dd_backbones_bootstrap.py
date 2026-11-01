"""DiffusionDet with other backbones: AP on VAL of the Real-Only arm (seed 21) with ResNet-50 (the final series, GPU
inference with 1 sampling step), ResNet-101 and Swin-Tiny, -Small and -Base, and the paired difference
"backbone - ResNet-50" with 95% CIs from a GROUP bootstrap over the group partition (same resamples as
seeds3_bootstrap.py). VAL only.

Predictions, under `predictions_dir("diffusiondet", "val")` (all with 1 sampling step):
  ResNet-50              steps1/exp07_real100_val_preds.json    (the runs of dd_steps_bootstrap.py)
  the other backbones    backbones/<tag>_val_preds.json         tag in res101, swintiny, swinsmall, swinbase
                         (the training runs themselves are the folders `exp07_real100__<tag>`)
Backbones without predictions are omitted.
Usage: python analysis/full_val/dd_backbones_bootstrap.py [--preds DIR] [--b 2000]
Outputs: analysis/full_val/out/dd_backbones.md and dd_backbones.json
"""

import argparse
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from eval.coco_tools import load_gt  # noqa: E402
from full_val_bootstrap import diff_row  # noqa: E402
from px_val_bootstrap import DIFF_HEAD, METRICS, evaluate, group_weights  # noqa: E402
from seeds3_bootstrap import OUT  # noqa: E402

BASE = "ResNet-50"
# name -> (folder under the predictions folder, file name without _val_preds.json)
RUNS = {
    BASE: ("steps1", "exp07_real100"),
    "ResNet-101": ("backbones", "res101"),
    "Swin-Tiny": ("backbones", "swintiny"),
    "Swin-Small": ("backbones", "swinsmall"),
    "Swin-Base": ("backbones", "swinbase"),
}


def compare(preds, gt, w, n_groups, b):
    """Evaluate the backbones found under `preds`; returns the markdown lines and the record for the json."""
    res = {
        n: evaluate(f, gt, Path(preds) / d, w)
        for n, (d, f) in RUNS.items()
        if (Path(preds) / d / f"{f}_val_preds.json").exists()
    }
    out = {"n_groups": n_groups, "B": b, "runs": {}, "pairs": []}
    md = [
        f"# DiffusionDet, Real-Only, seed 21: AP on VAL per backbone (GPU, 1 step; {n_groups} groups of the group "
        f"partition; group bootstrap, B = {b})",
        "",
        "| Backbone | AP | AP50 | AP75 |",
        "|---|---|---|---|",
    ]
    for n, r in res.items():
        o = r["official"]
        out["runs"][n] = {k: round(100 * v, 2) for k, v in o.items() if not isinstance(v, dict)}
        md.append(f"| {n} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75")) + " |")
    md += ["", f"## Backbone − {BASE} (same group resamples)", ""] + DIFF_HEAD
    for n in [x for x in res if x != BASE and BASE in res]:
        for m, _ in METRICS:
            md.append(
                diff_row(
                    f"{n} − {BASE}",
                    100 * (res[n][m + "0"] - res[BASE][m + "0"]),
                    100 * (res[n][m + "b"] - res[BASE][m + "b"]),
                    m,
                    out,
                    "pairs",
                )
            )
    return md, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--preds",
        default=str(paths.predictions_dir("diffusiondet", "val")),
        help="folder with the steps1/ and backbones/ subfolders",
    )
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    gt = load_gt(paths.annotations_path("val"))
    w, n_groups = group_weights(sorted(gt.getImgIds()), a.b)
    md, out = compare(a.preds, gt, w, n_groups, a.b)
    OUT.mkdir(exist_ok=True)
    (OUT / "dd_backbones.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / "dd_backbones.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
