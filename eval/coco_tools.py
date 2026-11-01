"""COCO evaluation (pycocotools) and per-image-weighted AP re-accumulation for paired bootstrap.

- `official(gt, dt, ids, names)`: standard COCOeval restricted to `ids` (AP, AP50, AP75, APs/m/l and
  per-class AP, AP50 and AP75).
- `Cache(gt, dt)`: runs COCOeval.evaluate() once and stores, per category, the matches of every
  detection (area "all", maxDets 100). `ap_batch_cpu` re-accumulates the COCO AP for any matrix of
  per-image weights (0/1 = subset; multinomial counts = bootstrap resample), reproducing
  COCOeval.accumulate().
"""
import collections
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import paths

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
MAX_DETS = 100
DETECTORS = ("rtdetrv2", "diffusiondet", "rfdetr")


def preds_file(detector, exp, split):
    """COCO results file `<exp>_<split>_preds.json` of an experiment for a detector and split."""
    return paths.predictions_dir(detector, split) / f"{exp}_{split}_preds.json"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def load_gt(path):
    return quiet(COCO, str(path))


def load_dets(path, gt, max_dets=MAX_DETS):
    """Read a COCO results JSON, check ids and keep the best `max_dets` detections per image."""
    dets = json.load(open(path, encoding="utf8"))
    bad_img = {d["image_id"] for d in dets} - set(gt.getImgIds())
    bad_cat = {d["category_id"] for d in dets} - set(gt.getCatIds())
    if bad_img or bad_cat:
        raise ValueError(f"{path}: {len(bad_img)} image_id and {len(bad_cat)} category_id not in the GT")
    by = collections.defaultdict(list)
    for d in dets:
        by[d["image_id"]].append(d)
    out = []
    for lst in by.values():
        out += sorted(lst, key=lambda d: -d["score"])[:max_dets]
    return quiet(gt.loadRes, out) if out else None


def official(gt, dt, img_ids, names):
    """Standard COCO metrics over `img_ids` (fractions 0-1) with per-class AP/AP50/AP75."""
    e = COCOeval(gt, dt, "bbox")
    e.params.imgIds = sorted(img_ids)
    quiet(e.evaluate); quiet(e.accumulate); quiet(e.summarize)
    out = {k: float(v) for k, v in zip(STAT_NAMES, e.stats[:6])}
    pr = e.eval["precision"][:, :, :, 0, 2]                 # T x R x K (area all, maxDets 100)
    for k, c in enumerate(e.params.catIds):
        for tag, sl in (("AP", slice(None)), ("AP50", slice(0, 1)), ("AP75", slice(5, 6))):
            p = pr[sl, :, k]
            p = p[p > -1]
            out[f"{tag}_{names[c]}"] = float(p.mean()) if p.size else float("nan")
    return out


class Cache:
    """Per-detection matches per category, used to re-accumulate the AP with per-image weights."""

    def __init__(self, gt, dt):
        e = COCOeval(gt, dt, "bbox")
        quiet(e.evaluate)
        p = e.params
        self.img_ids = list(p.imgIds)
        self.recThrs = p.recThrs
        n_img, n_area = len(self.img_ids), len(p.areaRng)
        self.cats = [self._cat(e, k, c, n_img, n_area) for k, c in enumerate(p.catIds)]

    @staticmethod
    def _cat(e, k, c, n_img, n_area):
        sc, im, tp, fp, npig = [], [], [], [], np.zeros(n_img)
        for i in range(n_img):
            ev = e.evalImgs[k * n_area * n_img + i]          # area "all"
            if ev is None:
                continue
            m, ig = ev["dtMatches"][:, :MAX_DETS], ev["dtIgnore"][:, :MAX_DETS]
            npig[i] = np.count_nonzero(ev["gtIgnore"] == 0)
            sc.append(np.asarray(ev["dtScores"][:MAX_DETS])); im.append(np.full(m.shape[1], i))
            tp.append(np.logical_and(m, ~ig)); fp.append(np.logical_and(~m.astype(bool), ~ig))
        t = len(e.params.iouThrs)
        if not sc:
            z = np.zeros((t, 0), np.float32)
            return dict(cat=c, img=np.zeros(0, int), tp=z, fp=z, npig=npig)
        o = np.argsort(-np.concatenate(sc), kind="mergesort")
        return dict(cat=c, img=np.concatenate(im)[o], npig=npig,
                    tp=np.concatenate(tp, 1)[:, o].astype(np.float32),
                    fp=np.concatenate(fp, 1)[:, o].astype(np.float32))


def _cat_ap(c, w, rec, t_idx, live):
    """AP of one category for each row of the weights `w` (B x I)."""
    n = w @ c["npig"]
    tp = c["tp"] if t_idx is None else c["tp"][t_idx]
    fp = c["fp"] if t_idx is None else c["fp"][t_idx]
    keep = live[c["img"]]
    tp, fp, img = tp[:, keep].astype(np.float64), fp[:, keep].astype(np.float64), c["img"][keep]
    anytp = np.where(tp.any(0))[0]
    cut = anytp[-1] + 1 if len(anytp) else 0                # the tail after the last TP does not change the AP
    tp, fp, img = tp[:, :cut], fp[:, :cut], img[:cut]
    t, nd = tp.shape
    out = np.zeros(len(w))
    for b in range(len(w)) if nd else []:
        wd = w[b, img]
        tps, fps = np.cumsum(tp * wd, -1), np.cumsum(fp * wd, -1)
        rc = tps / max(n[b], 1e-12)
        pr = np.maximum.accumulate((tps / (tps + fps + np.spacing(1)))[:, ::-1], axis=-1)[:, ::-1]
        for k in range(t):
            inds = np.searchsorted(rc[k], rec, side="left")
            out[b] += pr[k, inds[inds < nd]].sum() / (t * len(rec))
    out[n == 0] = np.nan
    return out


def ap_batch_cpu(cache, w, t_idx=None):
    """Mean AP over the categories with GT for each row of `w`; returns (B,) and (B x K)."""
    w = np.asarray(w, dtype=np.float64)
    live = w.any(0)
    per_cat = np.stack([_cat_ap(c, w, cache.recThrs, t_idx, live) for c in cache.cats], 1)
    with np.errstate(all="ignore"):
        return np.nanmean(per_cat, 1), per_cat
