"""Per-class AP of the Jetson engines and paired per-class differences with 95% CIs from a GROUP bootstrap (the same
resamples as ``int8_bootstrap.py``). VAL only, never TEST.

Same predictions and sets as ``int8_bootstrap.py`` (``<preds>/<name>_val_preds.json``).
Usage:
    python deploy_jetson/int8_per_class.py --set rtdetrv2_final --tag rt_final_15W
    python deploy_jetson/int8_per_class.py --set diffusiondet_final --tag dd_final_15W
Outputs: deploy_jetson/out/int8_per_class_<tag>.md and .json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis" / "proxy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from int8_bootstrap import OUT, SETS, available_runs  # noqa: E402


def per_class(exp, gt, preds, w):
    """Official per-class metrics, point per-class AP and its bootstrap resamples (B, classes) for one run."""
    from eval.coco_tools import Cache, ap_batch_cpu, load_dets, official
    dt = load_dets(Path(preds) / f"{exp}_val_preds.json", gt)
    cache = Cache(gt, dt)
    assert list(cache.img_ids) == sorted(gt.getImgIds())
    _, k0 = ap_batch_cpu(cache, np.ones((1, len(cache.img_ids))))
    _, kb = ap_batch_cpu(cache, w)
    return official(gt, dt, cache.img_ids, paths.CLASSES), k0[0], kb


def pair_section(title, names, a, b):
    """Markdown rows and JSON records of the per-class difference a - b, where a, b = (point AP, resamples)."""
    md = ["", f"## {title}, per class", "", "| Class | Difference | 95% CI |", "|---|---|---|"]
    rows = []
    for k, n in enumerate(names):
        d0 = 100 * (a[0][k] - b[0][k])
        lo, hi = np.nanpercentile(100 * (a[1][:, k] - b[1][:, k]), [2.5, 97.5])
        rows.append({"class": n, "diff": round(float(d0), 2), "ci": [round(float(lo), 2), round(float(hi), 2)]})
        md.append(f"| {n} | {d0:+.2f} | [{lo:+.2f}; {hi:+.2f}] |")
    return md, rows


def main():
    from eval.coco_tools import load_gt
    from px_val_bootstrap import group_weights
    from deploy_layout import boot_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=None, help="folder with <name>_val_preds.json (default: boot/<tag>)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--set", required=True, choices=list(SETS))
    a = ap.parse_args()
    preds = a.preds or boot_dir(a.tag)
    title, labels, pairs = SETS[a.set]
    gt = load_gt(paths.annotations_path("val"))
    cat_ids = sorted(gt.getCatIds())
    names = [paths.CLASSES[c] for c in cat_ids]
    n_gt = [len(gt.getAnnIds(catIds=[c])) for c in cat_ids]
    w, n_groups = group_weights(sorted(gt.getImgIds()), a.b)
    runs = available_runs(labels, preds)
    res = {}
    for e in runs:
        off, k0, kb = per_class(e, gt, preds, w)
        for k, n in enumerate(names):   # the order of Cache.cats is the order of catIds: checked against the official AP
            assert abs(k0[k] - off[f"AP_{n}"]) < 1e-6 or (np.isnan(k0[k]) and np.isnan(off[f"AP_{n}"])), (e, n)
        res[e] = (k0, kb)
    out = {"tag": a.tag, "set": a.set, "n_groups": n_groups, "B": a.b, "classes": names, "n_gt": n_gt,
           "ap": {e: [round(100 * float(v), 2) for v in res[e][0]] for e in runs}, "pairs": {}}
    md = [f"# {title}, mode {a.tag}: per-class AP (COCO) on VAL ({n_groups} groups of the group partition; "
          f"group bootstrap, B = {a.b})", "", "| Class | Instances | " + " | ".join(labels[e] for e in runs) + " |",
          "|---|---|" + "---|" * len(runs)]
    for k, n in enumerate(names):
        md.append(f"| {n} | {n_gt[k]} | " + " | ".join(f"{100 * res[e][0][k]:.2f}" for e in runs) + " |")
    for x, y in [q for q in pairs if q[0] in res and q[1] in res]:
        section, out["pairs"][f"{x} - {y}"] = pair_section(f"{labels[x]} − {labels[y]}", names, res[x], res[y])
        md += section
    OUT.mkdir(exist_ok=True)
    (OUT / f"int8_per_class_{a.tag}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    with open(OUT / f"int8_per_class_{a.tag}.json", "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
