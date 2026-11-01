"""Per-class analysis of the three detectors on TEST (group partition), with the frozen configuration.

Per detector and arm (mean of the 3 seeds 21, 22, 23):
  (a) AP, AP50 and AP75 per class with 95% CIs from a GROUP bootstrap (same resamples as seeds3_bootstrap.py) and
      the difference to Real-Only per class with CI and p;
  (b) AP by object size (COCO APs/APm/APl);
  (c) per-class error analysis at IoU 0.5 and a confidence threshold SELECTED ON VAL: for each run, the one that
      maximises the global (micro) F1 on VAL with greedy matching by score and class; it is applied unchanged to TEST;
  (d) confusion matrix of the model selected for deployment (chosen on VAL).
CPU only. Predictions: `predictions_dir(det, split) / <exp>[_s2|_s3]_<split>_preds.json` for val and test.
TEST is evaluated once, with the frozen configuration: set ALLOW_TEST=1 to run it.
Usage: ALLOW_TEST=1 python analysis/full_val/test_analysis.py [--b 2000] [--det rtdetrv2 ...]
Outputs: analysis/full_val/out/test_analysis.md and test_analysis.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis/proxy"))
from common import paths  # noqa: E402
from eval.coco_tools import DETECTORS, Cache, ap_batch_cpu, load_dets, load_gt, official, preds_file  # noqa: E402
from px_val_bootstrap import group_weights  # noqa: E402

ARMS = {"exp07_real100": "Real-Only", "exp08_real_synth_minor": "MF clean-bg",
        "exp06v11_synth100_real100": "Std copy-paste"}
SEEDS = ["", "_s2", "_s3"]
BASE = "exp07_real100"
# Model selected for deployment (chosen on VAL)
MAIN = {"rtdetrv2": "exp08_real_synth_minor_s3", "rfdetr": "exp07_real100", "diffusiondet": "exp06v11_synth100_real100_s3"}
CLASSES = paths.CLASSES
K = len(CLASSES)
IOU = 0.5
OUT = Path(__file__).resolve().parent / "out"


def lower_priority():
    """Lower the CPU priority of the process on Windows (no effect elsewhere)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
    except (ImportError, AttributeError, OSError):
        pass


def iou_mat(a, b):
    """IoU between xywh boxes (n,4) and (m,4)."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def by_image(gt, dt):
    """GT (without crowd) and detections per image: dict id -> (gt boxes, gt classes, det boxes, det classes,
    det scores), detections sorted by score."""
    g = {i: [] for i in gt.getImgIds()}
    for a in gt.dataset["annotations"]:
        if not a.get("iscrowd", 0):
            g[a["image_id"]].append((a["bbox"], a["category_id"]))
    d = {i: [] for i in g}
    for a in (dt.dataset["annotations"] if dt else []):
        d[a["image_id"]].append((a["bbox"], a["category_id"], a["score"]))
    out = {}
    for i in g:
        gb = np.array([x[0] for x in g[i]], float).reshape(-1, 4)
        gc = np.array([x[1] for x in g[i]], int)
        dd = sorted(d[i], key=lambda x: -x[2])
        db = np.array([x[0] for x in dd], float).reshape(-1, 4)
        dc = np.array([x[1] for x in dd], int)
        ds = np.array([x[2] for x in dd], float)
        out[i] = (gb, gc, db, dc, ds)
    return out


def match_same_class(gb, gc, db, dc, iou_thr=IOU):
    """Greedy matching by score and class; returns the matched gt index per detection (-1 = none)."""
    m = np.full(len(db), -1)
    if not len(db) or not len(gb):
        return m
    iou = iou_mat(db, gb)
    used = np.zeros(len(gb), bool)
    for j in range(len(db)):
        cand = np.where((gc == dc[j]) & ~used & (iou[j] >= iou_thr))[0]
        if len(cand):
            k = cand[np.argmax(iou[j, cand])]
            m[j] = k
            used[k] = True
    return m


def best_threshold(per_img):
    """Threshold that maximises the global F1 on VAL (IoU 0.5). Greedy matching by score does not depend on the
    threshold (the detections above it are a prefix), so matching is done once with all detections."""
    sc, tp, n_gt = [], [], 0
    for gb, gc, db, dc, ds in per_img.values():
        n_gt += len(gb)
        sc.append(ds)
        tp.append(match_same_class(gb, gc, db, dc) >= 0)
    sc, tp = np.concatenate(sc), np.concatenate(tp)
    o = np.argsort(-sc, kind="mergesort")
    sc, tp = sc[o], tp[o]
    ctp = np.cumsum(tp)
    n = np.arange(1, len(sc) + 1)
    f1 = 2 * ctp / (n + n_gt)                                  # 2TP / (2TP + FP + FN) = 2TP / (n_det + n_gt)
    last = np.r_[sc[1:] != sc[:-1], True]                      # cut only between different scores
    f1 = np.where(last, f1, -1)
    i = int(np.argmax(f1))
    return float(sc[i]), float(f1[i]), float(ctp[i] / (i + 1)), float(ctp[i] / n_gt)


def confusion(per_img, thr):
    """(K+1)x(K+1) count matrix, rows = GT (last = background), columns = prediction (last = background).
    1) hits of the same class; 2) unmatched detections against unmatched GT of another class (confusion);
    3) the rest are FP (background row) or FN (background column)."""
    cm = np.zeros((K + 1, K + 1), int)
    for gb, gc, db, dc, ds in per_img.values():
        keep = ds >= thr
        db, dc = db[keep], dc[keep]
        m = match_same_class(gb, gc, db, dc)
        used = np.zeros(len(gb), bool)
        used[m[m >= 0]] = True
        for j in np.where(m >= 0)[0]:
            cm[gc[m[j]], dc[j]] += 1
        rest = np.where(m < 0)[0]
        if len(rest) and (~used).any():
            iou = iou_mat(db[rest], gb)
            for r, j in enumerate(rest):
                cand = np.where(~used & (iou[r] >= IOU))[0]
                if len(cand):
                    k = cand[np.argmax(iou[r, cand])]
                    used[k] = True
                    cm[gc[k], dc[j]] += 1
                    m[j] = -2
                else:
                    cm[K, dc[j]] += 1
        else:
            for j in rest:
                cm[K, dc[j]] += 1
        for k in np.where(~used)[0]:
            cm[gc[k], K] += 1
    return cm


def cm_stats(cm):
    """FN %, FP % and confusions per class from the count matrix."""
    out = {}
    for c in range(K):
        n_gt, n_pred = cm[c].sum(), cm[:, c].sum()
        tp, missed = cm[c, c], cm[c, K]
        conf = n_gt - tp - missed
        fp_bg, fp_conf = cm[K, c], cm[:, c].sum() - tp - cm[K, c]
        out[CLASSES[c]] = {"n_gt": int(n_gt), "TP": int(tp), "FN_pct": 100 * (n_gt - tp) / max(n_gt, 1),
                           "missed_pct": 100 * missed / max(n_gt, 1), "confused_pct": 100 * conf / max(n_gt, 1),
                           "FP_pct": 100 * (n_pred - tp) / max(n_pred, 1), "FP_bg_pct": 100 * fp_bg / max(n_pred, 1),
                           "FP_conf_pct": 100 * fp_conf / max(n_pred, 1), "n_pred": int(n_pred)}
    return out


def eval_run(e, det, gt, w):
    """Per-class bootstrap AP, AP by size (TEST) and errors with the threshold selected on VAL."""
    t0 = time.time()
    dt = load_dets(preds_file(det, e, "test"), gt)
    cache = Cache(gt, dt)
    assert list(cache.img_ids) == sorted(gt.getImgIds())
    off = official(gt, dt, cache.img_ids, CLASSES)
    r = {"official": off, "ap": {}}
    ones = np.ones((1, len(cache.img_ids)))
    for m, t in (("AP", None), ("AP50", [0]), ("AP75", [5])):
        a0, pc0 = ap_batch_cpu(cache, ones, t)
        assert abs(a0[0] - off[m]) < 1e-6, (e, m, a0, off[m])
        ab, pcb = ap_batch_cpu(cache, w, t)
        r["ap"][m] = {"all0": float(a0[0]), "allb": ab, "cls0": pc0[0], "clsb": pcb}
    per_img_t = by_image(gt, dt)
    gv = load_gt(paths.annotations_path("val"))
    dv = load_dets(preds_file(det, e, "val"), gv)
    thr, f1, p, rc = best_threshold(by_image(gv, dv))
    r["thr"] = {"thr": thr, "F1_val": f1, "P_val": p, "R_val": rc}
    r["cm"] = confusion(per_img_t, thr)
    print(f"  {det} {e}: AP={100 * off['AP']:.2f} thr={thr:.3f} F1_val={f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return r


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def pval(db):
    return float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean())))


def arm_ap(res, runs, m):
    """Seed mean: point estimate and bootstrap, total and per class (in AP points)."""
    p0 = 100 * np.mean([res[e]["ap"][m]["cls0"] for e in runs], 0)
    pb = 100 * np.mean([res[e]["ap"][m]["clsb"] for e in runs], 0)
    t0 = 100 * np.mean([res[e]["ap"][m]["all0"] for e in runs])
    tb = 100 * np.mean([res[e]["ap"][m]["allb"] for e in runs], 0)
    return p0, pb, t0, tb


def arm_ap_blocks(ap_cls, a):
    """Total and per-class AP of arm `a` with CIs and, for non-base arms, the difference to Real-Only."""
    d = {"AP_total": {}, "per_class": {}}
    for m in ("AP", "AP50", "AP75"):
        p0, pb, t0, tb = ap_cls[a][m]
        d["AP_total"][m] = [round(float(t0), 2), ci(tb)]
        for c, name in enumerate(CLASSES):
            d["per_class"].setdefault(name, {})[m] = {"v": round(float(p0[c]), 2), "ci": ci(pb[:, c])}
            if a != BASE:
                q0, qb = ap_cls[BASE][m][0], ap_cls[BASE][m][1]
                db = pb[:, c] - qb[:, c]
                d["per_class"][name][m]["diff"] = {"v": round(float(p0[c] - q0[c]), 2), "ci": ci(db),
                                                   "p": round(pval(db), 4)}
        if a != BASE:
            db = tb - ap_cls[BASE][m][3]
            d["AP_total"][m] += [{"v": round(float(t0 - ap_cls[BASE][m][2]), 2), "ci": ci(db), "p": round(pval(db), 4)}]
    return d


def analyse_det(det, w):
    gt = load_gt(paths.annotations_path("test"))
    res = {e + s: eval_run(e + s, det, gt, w) for e in ARMS for s in SEEDS}
    runs = {a: [a + s for s in SEEDS] for a in ARMS}
    o = {"arms": {}, "main": {"run": MAIN[det]}, "thresholds": {}}
    ap_cls = {a: {m: arm_ap(res, runs[a], m) for m in ("AP", "AP50", "AP75")} for a in ARMS}
    for a in ARMS:
        d = {**arm_ap_blocks(ap_cls, a), "size": {}, "errors": {}}
        for k in ("APs", "APm", "APl"):
            v = 100 * np.array([res[e]["official"][k] for e in runs[a]])
            d["size"][k] = {"v": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2)}
        stats = [cm_stats(res[e]["cm"]) for e in runs[a]]
        for name in CLASSES:
            d["errors"][name] = {k: round(float(np.mean([s[name][k] for s in stats])), 2) for k in stats[0][name]}
        cm_mean = np.mean([res[e]["cm"] for e in runs[a]], 0)
        d["confusion_mean_counts"] = np.round(cm_mean, 1).tolist()
        d["thresholds"] = {e: {k: round(v, 4) for k, v in res[e]["thr"].items()} for e in runs[a]}
        o["arms"][a] = d
    mr = res[MAIN[det]]
    o["main"].update({"thr": round(mr["thr"]["thr"], 4), "AP": round(100 * mr["official"]["AP"], 2),
                      "confusion_counts": mr["cm"].tolist(), "stats": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                                                                        for k, v in cm_stats(mr["cm"]).items()}})
    return o


def md_thresholds(o):
    md = ["Confidence threshold selected on VAL (maximum F1) per run:", "",
          "| Run | Threshold | F1 on VAL | P on VAL | R on VAL |", "|---|---|---|---|---|"]
    for d in o["arms"].values():
        md += [f"| {e} | {v['thr']:.3f} | {v['F1_val']:.3f} | {v['P_val']:.3f} | {v['R_val']:.3f} |"
               for e, v in d["thresholds"].items()]
    return md


def md_ap_tables(det, o):
    md = []
    for m in ("AP", "AP50", "AP75"):
        md += ["", f"### {det}: {m} per class (mean of 3 seeds; 95% CI from the group bootstrap; "
                   f"Δ against Real-Only with CI and p)", "",
               "| Class | " + " | ".join(ARMS[a] for a in ARMS) + " | Δ MF clean-bg | Δ Std copy-paste |",
               "|---|" + "---|" * 5]
        for name in CLASSES + ["Total"]:
            cells, dif = [], []
            for a in ARMS:
                d = o["arms"][a]
                e = d["AP_total"][m] if name == "Total" else d["per_class"][name][m]
                v, c = (e[0], e[1]) if name == "Total" else (e["v"], e["ci"])
                cells.append(f"{v:.2f} [{c[0]:.2f}; {c[1]:.2f}]")
                if a != BASE:
                    x = e[2] if name == "Total" else e["diff"]
                    dif.append(f"{x['v']:+.2f} [{x['ci'][0]:+.2f}; {x['ci'][1]:+.2f}] (p = {x['p']:.3f})")
            md.append(f"| {name} | " + " | ".join(cells + dif) + " |")
    return md


def md_size_errors(det, o):
    md = ["", f"### {det}: AP by size (mean of 3 seeds ± SD across seeds)", "",
          "| Arm | APs | APm | APl |", "|---|---|---|---|"]
    for a, d in o["arms"].items():
        md.append(f"| {ARMS[a]} | " + " | ".join(f"{d['size'][k]['v']:.2f} ± {d['size'][k]['sd']:.2f}"
                                                 for k in ("APs", "APm", "APl")) + " |")
    md += ["", f"### {det}: per-class errors on TEST (threshold selected on VAL, IoU 0.5; mean of 3 seeds)", "",
           "FN % = GT without a correct detection (no box + confused with another class); FP % = detections of the "
           "class that are not hits (background + confusion).", "",
           "| Arm | Class | n GT | FN % | of which no box % | of which confused % | FP % | FP background % | "
           "FP confusion % |", "|---|---|---|---|---|---|---|---|---|"]
    for a, d in o["arms"].items():
        for name in CLASSES:
            s = d["errors"][name]
            md.append(f"| {ARMS[a]} | {name} | {s['n_gt']:.0f} | {s['FN_pct']:.1f} | {s['missed_pct']:.1f} | "
                      f"{s['confused_pct']:.1f} | {s['FP_pct']:.1f} | {s['FP_bg_pct']:.1f} | {s['FP_conf_pct']:.1f} |")
    return md


def md_confusions(det, o):
    md = ["", f"### {det}: most frequent confusions (GT -> predicted), % of the GT of the class, mean of 3 seeds", "",
          "| Arm | GT -> predicted | % of GT | mean count |", "|---|---|---|---|"]
    for a, d in o["arms"].items():
        cm = np.array(d["confusion_mean_counts"])
        pairs = sorted(((cm[i, j], i, j) for i in range(K) for j in range(K) if i != j), reverse=True)[:5]
        for n, i, j in pairs:
            md.append(f"| {ARMS[a]} | {CLASSES[i]} -> {CLASSES[j]} | {100 * n / cm[i].sum():.1f} | {n:.1f} |")
    mn = o["main"]
    md += ["", f"### {det}: deployed model ({mn['run']}), threshold {mn['thr']:.3f}, AP {mn['AP']:.2f}", "",
           "Count matrix (rows = GT, columns = predicted; last = background).", "",
           "| GT \\ pred | " + " | ".join(CLASSES + ["Background"]) + " |", "|---|" + "---|" * (K + 1)]
    for i, row in enumerate(mn["confusion_counts"]):
        md.append(f"| {(CLASSES + ['Background'])[i]} | " + " | ".join(str(x) for x in row) + " |")
    return md


def md_det(det, o):
    return ["", f"## {det}", ""] + md_thresholds(o) + md_ap_tables(det, o) + md_size_errors(det, o) + md_confusions(det, o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--det", nargs="*", default=list(DETECTORS), choices=list(DETECTORS))
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    if os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    lower_priority()
    gt = load_gt(paths.annotations_path("test"))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    out = {"n_img": len(ids), "n_groups": n_groups, "B": a.b, "iou_errors": IOU, "dets": {}}
    md = [f"# Per-class analysis on TEST (n = {len(ids)} images, {n_groups} groups of the group partition; "
          f"group bootstrap, B = {a.b}; 3 seeds per arm)", "",
          "Confidence threshold of the errors: per run, the one that maximises the global (micro) F1 on VAL at IoU 0.5 "
          "(greedy matching by score and class), applied unchanged to TEST. The AP values do not depend on the threshold."]
    for det in a.det:
        out["dets"][det] = analyse_det(det, w)
        md += md_det(det, out["dets"][det])
    OUT.mkdir(exist_ok=True)
    sfx = "" if len(a.det) == 3 else "_" + "_".join(a.det)
    (OUT / f"test_analysis{sfx}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / f"test_analysis{sfx}.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
