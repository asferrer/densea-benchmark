"""Shared helpers of the copy-paste gap diagnosis: weighted AUC, grouped-CV AUC, formatting and tables."""
import numpy as np

import variant_features as gf

dg = gf.dg
SEEDS = (dg.SEED, dg.SEED + 1, dg.SEED + 2)
CLASSES = dg.CLASSES
FAMILIES = {
    "sharpness (lapvar_native, lapvar64, hf_ratio)": ["lapvar_native", "lapvar64", "hf_ratio"],
    "noise in flat regions (noise_flat)": ["noise_flat"], "JPEG blocks (block)": ["block"],
    "color (sat, val, hue_cos, hue_sin)": ["sat", "val", "hue_cos", "hue_sin"],
    "contrast (contrast_dE, l_std)": ["contrast_dE", "l_std"],
    "geometry (rel_size, log_aspect, cx, cy, border_dist, min_side)":
        ["rel_size", "log_aspect", "cx", "cy", "border_dist", "min_side"]}
STAT_KEYS = [k for ks in FAMILIES.values() for k in ks]


def wauc(score, y, cls, sel):
    """AUC of `score` for the positives `sel` (bool) against ALL the negatives reweighted to the class mix."""
    from sklearn.metrics import roc_auc_score
    pos_c = {c: int(((cls == c) & sel).sum()) for c in CLASSES}
    w = np.zeros(len(y))
    for c in CLASSES:
        neg = (y == 0) & (cls == c)
        w[neg] = pos_c[c] / max(neg.sum(), 1)
    use = sel | ((y == 0) & (w > 0))
    if sel.sum() < 5 or (use & (y == 0)).sum() < 5:
        return float("nan"), float("nan")
    w[sel] = 1.0
    a = roc_auc_score(y[use], score[use], sample_weight=w[use])
    n1, n0 = int(sel.sum()), w[use & (y == 0)].sum() ** 2 / (w[use & (y == 0)] ** 2).sum()
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return float(a), float(se)


def cv_auc(x, y, groups):
    """Mean AUC (3 seeds, grouped CV) and mean out-of-fold probabilities."""
    runs = [dg.cv_classify(x, y, groups=groups, seed=s) for s in SEEDS]
    return {"auc": float(np.mean([r["auc"] for r in runs])), "std": float(np.std([r["auc"] for r in runs])),
            "oof": np.mean([r["oof"] for r in runs], axis=0)}


def fmt(a, se=None):
    if a != a:
        return "n/a"
    return f"{a:.3f}" + (f" ± {se:.3f}" if se is not None and se == se else "")


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def touches(items, tol=3):
    """True if the box touches the image border (within tol px): object placed against the border or cut by it."""
    return np.array([i["bbox"][0] <= tol or i["bbox"][1] <= tol or i["bbox"][0] + i["bbox"][2] >= i["w"] - tol
                     or i["bbox"][1] + i["bbox"][3] >= i["h"] - tol for i in items])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
