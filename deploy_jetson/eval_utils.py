"""COCO evaluation and timing statistics shared by the VAL evaluation scripts."""
import numpy as np


def coco_ap(val_json, preds, ids):
    """AP, AP50 and AP75 (0-1 scale) of COCO detections on the images ``ids`` of ``val_json``."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(val_json))
    ev = COCOeval(gt, gt.loadRes(preds), "bbox")
    ev.params.imgIds = ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2]}


def time_stats(seconds, skip=20):
    """Mean, median and p99 in ms of per-image times in seconds, discarding the first ``skip`` (warm-up)."""
    t = 1000 * np.array(seconds[skip:])
    return {"mean": float(t.mean()), "median": float(np.median(t)), "p99": float(np.percentile(t, 99))}
