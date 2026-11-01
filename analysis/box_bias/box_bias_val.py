"""Box-size bias of RT-DETRv2 on VAL (never TEST) when training with the clean-background synthetic batch (v9), whose
boxes include a margin around the object.

1. Curves from log.txt (`test_coco_eval_bbox` = COCO metrics on VAL; `test` is the name used by the RT-DETR repo).
2. Size bias of the predicted boxes on VAL (best.pth, CPU inference with infer_val_cpu.py): greedy matching by score
   (IoU >= 0.5, same class), pred/gt ratios and shift per side, 95% CIs from the group bootstrap over the group
   partition; AP recomputed with pycocotools; post-hoc correction (mean bias estimated on one half of VAL split by
   groups, evaluated on the other).
Predictions: `predictions_dir("rtdetrv2", "val") / <exp>_val_preds.json` and `val_ids.json` (see --preds).
Usage: python analysis/box_bias/box_bias_val.py [--preds DIR] [--workers 6] [--k-splits 5]
Outputs: analysis/box_bias/out/box_bias_val.json, box_bias_val.md and curves.csv
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
from analysis.box_bias import box_bias_core as C  # noqa: E402
from common import paths  # noqa: E402

RUNS = paths.RUNS_DIR / "rtdetrv2"
EXPS = ["exp07_real100", "exp06_synth100_real100", "exp08_real_synth_minor", "exp09_real_synth_invprop"]
REF = EXPS[0]
VAL_JSON = paths.annotations_path("val")
OUT = Path(__file__).resolve().parent / "out"
SCORE_THR = 0.3          # main score threshold of the predictions used for matching
SENS_THR = (0.1, 0.5)    # threshold sensitivity
B_BOOT = 2000
K_SPLITS = 5


def lower_priority():
    """Run the process at the lowest CPU priority (Windows idle class, nice 19 elsewhere)."""
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)


# ---------------------------------------------------------------- curves from log.txt
def read_curve(exp):
    rows = [json.loads(line) for line in open(RUNS / exp / "log.txt", encoding="utf8") if line.strip()]
    return {r["epoch"] + 1: [100 * x for x in r["test_coco_eval_bbox"]] for r in rows}


def best_epoch(curve):
    return max(curve, key=lambda e: curve[e][0])


# ---------------------------------------------------------------- data and matching
def load_val():
    """VAL images by id, group of each image (`group` field of the annotations) and GT boxes by image."""
    val = json.load(open(VAL_JSON, encoding="utf8"))
    imgs = {i["id"]: i for i in val["images"]}
    grp = {i: img["group"] for i, img in imgs.items()}
    gts = {}
    for a in val["annotations"]:
        x, y, w, h = a["bbox"]
        gts.setdefault(a["image_id"], []).append((x, y, x + w, y + h, a["category_id"], a["area"]))
    return imgs, grp, gts


def load_preds(path):
    """Format of COCO.loadRes(ndarray): [img, x, y, w, h, score, cls]."""
    d = json.load(open(path))
    return np.array([[p["image_id"], *p["bbox"], p["score"], p["category_id"]] for p in d], float)


def size_bin(area):
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


def rel_bin(g, img):
    s = np.sqrt((g[2] - g[0]) * (g[3] - g[1]) / (img["width"] * img["height"]))
    return "<5 %" if s < 0.05 else ("5-15 %" if s < 0.15 else ">15 %")


def match_all(preds, imgs, grp, gts, thr):
    """Dict of arrays with one row per pred-GT pair (predictions with score >= thr)."""
    cols = {k: [] for k in ["img", "grp", "cls", "size", "rsize", "score", "iou", "gw", "gh", *C.METRICS]}
    sel = preds[preds[:, 5] >= thr]
    sel = sel[np.argsort(sel[:, 0], kind="stable")]
    ids = sorted(imgs)
    starts = np.searchsorted(sel[:, 0], ids, "left")
    ends = np.searchsorted(sel[:, 0], ids, "right")
    for iid, s, e in zip(ids, starts, ends):
        g = np.array(gts.get(iid, []), float).reshape(-1, 6)
        p = sel[s:e]
        pb = np.c_[p[:, 1], p[:, 2], p[:, 1] + p[:, 3], p[:, 2] + p[:, 4]]
        for pi, gi, iou in C.greedy_match(pb, p[:, 6], p[:, 5], g[:, :4], g[:, 4], 0.5):
            vals = {"img": iid, "grp": grp[iid], "cls": int(g[gi, 4]), "size": size_bin(g[gi, 5]),
                    "rsize": rel_bin(g[gi, :4], imgs[iid]), "score": p[pi, 5], "iou": iou,
                    "gw": g[gi, 2] - g[gi, 0], "gh": g[gi, 3] - g[gi, 1], **C.side_bias(pb[pi], g[gi, :4])}
            for k, v in vals.items():
                cols[k].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def gt_counts(imgs, gts):
    out = {"size": {}, "rsize": {}}
    for iid, lst in gts.items():
        for g in lst:
            for k, v in (("size", size_bin(g[5])), ("rsize", rel_bin(g[:4], imgs[iid]))):
                out[k][v] = out[k].get(v, 0) + 1
    return out


def add_group_index(m, gpos):
    m["gi"] = np.array([gpos[g] for g in m["grp"]], int)
    return m


# ---------------------------------------------------------------- AP with pycocotools
_W = {}


def _init_worker(gt_path, preds, low_priority=True):
    import contextlib
    import io

    from pycocotools.coco import COCO
    if low_priority:
        lower_priority()
    with contextlib.redirect_stdout(io.StringIO()):
        _W["gt"] = COCO(gt_path)
    _W["preds"] = preds


def coco_stats(gt, arr, img_ids):
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(arr), "bbox")
        ev.params.imgIds = sorted(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return [100 * float(x) for x in ev.stats]


def eval_job(job):
    """job = (exp, ids, shift|None, mode): applies the correction to the boxes and evaluates on `ids`."""
    exp, ids, shift, mode = job
    a = _W["preds"][exp]
    a = a[np.isin(a[:, 0], list(ids))].copy()
    if shift is not None:
        xyxy = np.c_[a[:, 1], a[:, 2], a[:, 1] + a[:, 3], a[:, 2] + a[:, 4]]
        n = C.shrink_boxes(xyxy, shift, mode)
        a[:, 1], a[:, 2], a[:, 3], a[:, 4] = n[:, 0], n[:, 1], n[:, 2] - n[:, 0], n[:, 3] - n[:, 1]
    return coco_stats(_W["gt"], a, ids)


# ---------------------------------------------------------------- bootstrap and correction
def diff_ci(ms, counts, metric, kind_stat, exp):
    """95% CI of the difference exp - REF of the mean/median of `metric`, with the same resamples."""
    bs = [C.weighted_stat(ms[e][metric], ms[e]["gi"], counts, kind_stat) for e in (exp, REF)]
    return [float(x) for x in np.percentile(bs[0] - bs[1], [2.5, 97.5])]


def rows_of(m, gset):
    k = np.isin(m["grp"], list(gset))
    return {x: m[x][k] for x in ["gw", "gh", "out_l", "out_t", "out_r", "out_b"]}


def correction_jobs(ms, groups_all, imgs_by_group, k_splits):
    """One job per (random split of groups, direction, exp, variant)."""
    jobs = []
    for s in range(k_splits):
        perm = np.random.default_rng(100 + s).permutation(groups_all)
        halves = [set(perm[: len(perm) // 2].tolist()), set(perm[len(perm) // 2:].tolist())]
        for fit, ev in ((0, 1), (1, 0)):
            ids = {i for g in halves[ev] for i in imgs_by_group[g]}
            for e in EXPS:
                jobs.append({"split": s, "fit": fit, "exp": e, "mode": "base", "shift": None, "ids": ids})
                for mode in ("rel", "px"):
                    shift = C.fit_shift(rows_of(ms[e], halves[fit]), mode)
                    jobs.append({"split": s, "fit": fit, "exp": e, "mode": mode, "shift": shift, "ids": ids})
    return jobs


def run_correction(jobs, gt_path, preds, workers):
    args = [(j["exp"], j["ids"], j["shift"], j["mode"]) for j in jobs]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(gt_path, preds)) as ex:
        res = list(ex.map(eval_job, args, chunksize=2))
    return [{k: v for k, v in j.items() if k != "ids"} | {"stats": r} for j, r in zip(jobs, res)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=str(paths.predictions_dir("rtdetrv2", "val")),
                    help="folder with <exp>_val_preds.json and val_ids.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k-splits", type=int, default=K_SPLITS)
    a = ap.parse_args()
    lower_priority()
    from analysis.box_bias import report
    imgs, grp, gts = load_val()
    ids_infer = set(json.load(open(Path(a.preds) / "val_ids.json")))
    preds = {e: load_preds(Path(a.preds) / f"{e}_val_preds.json") for e in EXPS}
    curves = {e: read_curve(e) for e in EXPS}
    data = {"curves": curves, "n_imgs": len(ids_infer), "n_val": len(imgs), "score_thr": SCORE_THR}
    _init_worker(str(VAL_JSON), preds, low_priority=False)
    data["calib"] = {e: {"mine": coco_stats(_W["gt"], preds[e], ids_infer),
                         "log": curves[e][best_epoch(curves[e])], "best_epoch": best_epoch(curves[e])}
                     for e in EXPS}
    glist = sorted(set(grp.values()))
    gpos = {g: i for i, g in enumerate(glist)}
    counts = C.boot_counts(len(glist), B_BOOT, 7)
    ms = {e: add_group_index(match_all(preds[e], imgs, grp, gts, SCORE_THR), gpos) for e in EXPS}
    sens = {t: {e: add_group_index(match_all(preds[e], imgs, grp, gts, t), gpos) for e in EXPS}
            for t in SENS_THR}
    data["bias"] = report.bias_blocks(ms, sens, counts)
    data["bias"]["diff"] = {e: {m: {s: diff_ci(ms, counts, m, s, e) for s in ("mean", "median")}
                                for m in ("ratio_area", "ratio_w", "ratio_h", "iou", "abslog")} for e in EXPS[1:]}
    data["gt_counts"] = gt_counts(imgs, gts)
    ibg = {}
    for i, g in grp.items():
        ibg.setdefault(g, []).append(i)
    jobs = correction_jobs(ms, glist, ibg, a.k_splits)
    data["corr"] = run_correction(jobs, str(VAL_JSON), preds, a.workers)
    OUT.mkdir(exist_ok=True)
    json.dump(data, open(OUT / "box_bias_val.json", "w", encoding="utf8"), indent=1, default=float)
    report.write(data, OUT)


if __name__ == "__main__":
    main()
