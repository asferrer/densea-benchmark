"""Core of the box-bias analysis (no I/O): matching, bias per side, group bootstrap and post-hoc correction.
Boxes are xyxy unless stated otherwise."""
import numpy as np

METRICS = ["ratio_w", "ratio_h", "ratio_area", "out_l", "out_t", "out_r", "out_b",
           "rel_l", "rel_t", "rel_r", "rel_b"]


def iou_matrix(p, g):
    """IoU between xyxy boxes p (n,4) and g (m,4)."""
    p, g = np.asarray(p, float).reshape(-1, 4), np.asarray(g, float).reshape(-1, 4)
    lt = np.maximum(p[:, None, :2], g[None, :, :2])
    rb = np.minimum(p[:, None, 2:], g[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-12)


def greedy_match(pbox, pcls, pscore, gbox, gcls, thr=0.5):
    """Greedy matching by score: each prediction (from highest to lowest score) takes the free GT of its class
    with the highest IoU if IoU >= thr. Returns a list of (pred_index, gt_index, iou)."""
    if len(pbox) == 0 or len(gbox) == 0:
        return []
    iou = iou_matrix(pbox, gbox)
    pcls, gcls = np.asarray(pcls), np.asarray(gcls)
    iou = np.where(pcls[:, None] == gcls[None, :], iou, -1.0)
    used, out = np.zeros(len(gbox), bool), []
    for i in np.argsort(-np.asarray(pscore), kind="stable"):
        cand = np.where(used, -1.0, iou[i])
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            used[j] = True
            out.append((int(i), j, float(cand[j])))
    return out


def side_bias(p, g):
    """Bias of a predicted box p with respect to its GT g (xyxy). out_* > 0: the prediction protrudes from the
    GT on that side (px); rel_*: the same divided by the GT width (l, r) or height (t, b)."""
    pw, ph, gw, gh = p[2] - p[0], p[3] - p[1], g[2] - g[0], g[3] - g[1]
    out = {"out_l": g[0] - p[0], "out_t": g[1] - p[1], "out_r": p[2] - g[2], "out_b": p[3] - g[3]}
    return {"ratio_w": pw / gw, "ratio_h": ph / gh, "ratio_area": pw * ph / (gw * gh), **out,
            "rel_l": out["out_l"] / gw, "rel_t": out["out_t"] / gh,
            "rel_r": out["out_r"] / gw, "rel_b": out["out_b"] / gh}


def boot_counts(n_groups, B, seed):
    """Matrix (B, n_groups) of multinomial counts: group resampling with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.ones(n_groups) / n_groups, size=B)


def weighted_stat(val, grp, counts, kind):
    """Mean or median of `val` for each resample (row of `counts`); grp = group of each value."""
    val, grp = np.asarray(val, float), np.asarray(grp)
    if kind == "mean":
        w = counts[:, grp]
        return (w * val).sum(1) / np.maximum(w.sum(1), 1)
    o = np.argsort(val, kind="stable")
    w = counts[:, grp[o]]
    cw = np.cumsum(w, 1)
    idx = (cw < cw[:, -1:] / 2).sum(1)
    return val[o][np.minimum(idx, len(val) - 1)]


def summarize(val, grp, counts):
    """Mean and median with 95% CIs (percentiles) from the group bootstrap."""
    val = np.asarray(val, float)
    if len(val) == 0:
        return None
    out = {"n": int(len(val))}
    for kind, f in (("mean", np.mean), ("median", np.median)):
        b = weighted_stat(val, grp, counts, kind)
        out[kind] = float(f(val))
        out[kind + "_ci"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
    return out


def fit_shift(rows, mode):
    """Mean bias per side estimated on `rows` (dict of arrays with out_* and rel_*). For mode='rel' it is
    estimated relative to the dimension of the PREDICTION (it is applied to boxes without GT): out / (dim_gt + out_l + out_r)."""
    if mode == "px":
        return {s: float(np.mean(rows["out_" + s])) for s in "ltrb"}
    pw = (rows["gw"] + rows["out_l"] + rows["out_r"])
    ph = (rows["gh"] + rows["out_t"] + rows["out_b"])
    dim = {"l": pw, "r": pw, "t": ph, "b": ph}
    return {s: float(np.mean(rows["out_" + s] / dim[s])) for s in "ltrb"}


def shrink_boxes(boxes, shift, mode, min_size=1.0):
    """Shrink xyxy boxes (n,4) by subtracting `shift` per side (px or fraction of the box dimension)."""
    b = np.asarray(boxes, float).copy()
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    sx, sy = (w, h) if mode == "rel" else (1.0, 1.0)
    out = b.copy()
    out[:, 0] += shift["l"] * sx
    out[:, 1] += shift["t"] * sy
    out[:, 2] -= shift["r"] * sx
    out[:, 3] -= shift["b"] * sy
    bad = (out[:, 2] - out[:, 0] < min_size) | (out[:, 3] - out[:, 1] < min_size)
    out[bad] = b[bad]
    return out
