"""Overfitting diagnostic: training loss, VAL AP per epoch and TEST AP of the intermediate checkpoints
(run/launch_diag.py) of each run.

DIAGNOSTIC ONLY: the reported model is the one selected on VAL (its TEST predictions are
`predictions_dir(detector, "test") / <exp>_test_preds.json`). The TEST AP of the intermediate checkpoints does
not change that selection.
Training logs are read from `RUNS_DIR/<detector>/<exp>` (log.txt, eval_history.jsonl, schedule.json) and the
intermediate-checkpoint predictions from `RUNS_DIR/<detector>/<exp>/eval_diag/<tag>/`.
Outputs (eval/out/diag/): <det>_<exp>.csv, <det>_<exp>.png and SUMMARY.md
TEST is evaluated once, with the frozen configuration: set ALLOW_TEST=1 to run it.
Usage: ALLOW_TEST=1 python eval/diag_curves.py [--only exp07_real100 ...]
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from eval.coco_tools import DETECTORS, load_dets, load_gt, official, preds_file  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "diag"
DIAG_PRED = {"rtdetrv2": "*/coco_predictions.json", "rfdetr": "coco_predictions.json",
             "diffusiondet": "inference/densea_test/coco_instances_results.json"}


def json_lines(path):
    for line in open(path, encoding="utf8", errors="ignore"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def detr_curve(run, ap_key):
    """RT-DETRv2 and RF-DETR: log.txt with one JSON line per epoch (0-based epoch)."""
    rows = {}
    for r in json_lines(run / "log.txt"):
        if isinstance(r.get("epoch"), int) and isinstance(r.get(ap_key), list):
            rows[r["epoch"] + 1] = (r.get("train_loss"), r[ap_key][0])
    return rows


def dd_curve(run):
    """DiffusionDet: total_loss from log.txt and bbox/AP (0-100) from eval_history.jsonl, per epoch."""
    ipe = json.load(open(run / "schedule.json"))["SOLVER.CHECKPOINT_PERIOD"] // 10
    loss = {}
    for line in open(run / "log.txt", encoding="utf8", errors="ignore"):
        m = re.search(r"iter: (\d+)\s.*?total_loss: ([\d.]+)", line)
        if m:
            loss.setdefault(math.ceil((int(m.group(1)) + 1) / ipe), {})[int(m.group(1))] = float(m.group(2))
    rows = {}
    hist = run / "eval_history.jsonl"
    for r in (json_lines(hist) if hist.exists() else []):
        ep = (r["iteration"] + 1) // ipe
        tl = loss.get(ep)
        rows[ep] = (sum(tl.values()) / len(tl) if tl else None, r["bbox/AP"] / 100)
    return rows


def curve(det, run):
    if det == "rtdetrv2":
        return detr_curve(run, "test_coco_eval_bbox")
    if det == "rfdetr":
        return detr_curve(run, "ema_test_coco_eval_bbox")
    return dd_curve(run)


def test_ap(gt, ids, files):
    """AP on `ids` of the first existing predictions file, or None."""
    files = [f for f in files if Path(f).exists()]
    return official(gt, load_dets(files[0], gt), ids, paths.CLASSES)["AP"] if files else None


def diag_points(det, run, gt, ids, last_ep):
    pts = {}
    for d in sorted((run / "eval_diag").glob("*/DONE")):
        tag = d.parent.name
        ep = last_ep if tag == "last" else int(tag[2:])
        ap = test_ap(gt, ids, sorted(d.parent.glob(DIAG_PRED[det])))
        if ap is not None:
            pts[ep] = ap
    return pts


def plot(rows, pts, best_ep, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = sorted(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(eps, [rows[e][0] for e in eps], color="tab:gray")
    a1.set(xlabel="Epoch", ylabel="Training loss", title=title)
    a2.plot(eps, [100 * rows[e][1] for e in eps], label="VAL (selection)")
    if pts:
        a2.plot(sorted(pts), [100 * pts[e] for e in sorted(pts)], "o--", label="TEST (diagnostic)")
    a2.axvline(best_ep, color="tab:red", lw=0.8, ls=":", label=f"best on VAL (epoch {best_ep})")
    a2.set(xlabel="Epoch", ylabel="AP [.5:.95] (%)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyse(det, exp, run, gt, ids):
    rows = curve(det, run)
    if not rows:
        return None
    best_ep = max(rows, key=lambda e: rows[e][1])
    pts = diag_points(det, run, gt, ids, max(rows))
    with open(OUT / f"{det}_{exp}.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_AP", "test_AP_diag"])
        for e in sorted(rows):
            w.writerow([e, rows[e][0], round(100 * rows[e][1], 2),
                        round(100 * pts[e], 2) if e in pts else ""])
    plot(rows, pts, best_ep, f"{det} {exp}", OUT / f"{det}_{exp}.png")
    sel = test_ap(gt, ids, [preds_file(det, exp, "test")])
    top = max(pts, key=pts.get) if pts else None
    return {"det": det, "exp": exp, "epochs": max(rows), "best_ep": best_ep,
            "val_best": 100 * rows[best_ep][1], "test_sel": None if sel is None else 100 * sel,
            "diag_top_ep": top, "diag_top": None if top is None else 100 * pts[top]}


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def summary(res):
    md = ["# Overfitting diagnostic (diagnostic only: the selection is made on VAL)", "",
          "| Detector | Experiment | Epochs | Best epoch on VAL | VAL AP | TEST AP of the selected model | "
          "Best TEST AP among intermediate checkpoints (epoch) |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['det']} | {r['exp']} | {r['epochs']} | {r['best_ep']} | {fmt(r['val_best'])} | "
           f"{fmt(r['test_sel'])} | {fmt(r['diag_top'])} ({r['diag_top_ep'] or '-'}) |" for r in res]
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these experiment names")
    args = ap.parse_args()
    if os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt(paths.annotations_path("test"))
    ids = gt.getImgIds()
    res = []
    for det in DETECTORS:
        for run in sorted((paths.RUNS_DIR / det).glob("exp*")):
            if (args.only and run.name not in args.only) or not (run / "log.txt").exists():
                continue
            r = analyse(det, run.name, run, gt, ids)
            if r:
                res.append(r); print("analysed", det, run.name, flush=True)
    (OUT / "SUMMARY.md").write_text("\n".join(summary(res)) + "\n", encoding="utf8")
    print("written", OUT)


if __name__ == "__main__":
    main()
