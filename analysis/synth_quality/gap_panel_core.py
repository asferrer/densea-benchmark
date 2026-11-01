"""Pure metrics (numpy/sklearn) of the real-vs-synthetic gap panel. No images, no network, no GPU.

Conventions: `real` is the reference set and `fake` the synthetic one (n x d). Everything neighbor-based uses k = 5.
  * FD: Frechet between Gaussians (equals `domain_gap.frechet_distance`; here via an n x n SVD, much faster).
  * KID: unbiased MMD^2 with the cubic polynomial kernel (Binkowski 2018), on the raw features.
  * CMMD: MMD with an RBF kernel on L2-normalized embeddings (Jayasumana 2024). sigma = 10, scale x1000. The official
    code uses the BIASED estimator (it includes the diagonal); that one is given here and, separately, the unbiased one.
  * Precision/recall (Kynkaanniemi 2019) and density/coverage (Naeem 2020), on L2-normalized vectors.
  * TSTR: multiclass logistic regression trained on synthetic crops and evaluated on real crops.
Orientation: SENSE[m] = +1 if "higher = better synthetic quality", -1 if "higher = larger gap".
"""
import numpy as np

K_NN = 5
CMMD_SIGMA, CMMD_SCALE = 10.0, 1000.0
SENSE = {"auc": -1, "fd": -1, "kid": -1, "cmmd": -1, "cmmd_unb": -1, "precision": 1, "recall": 1, "density": 1,
         "coverage": 1, "tstr_bacc": 1, "tstr_recall": 1, "nn_real": 1, "vendi": 1, "nn_dist": 1}


def sense_of(metric):
    """Orientation of a metric, ignoring the reference suffix ('@val')."""
    return SENSE[metric.split("@")[0]]


def l2n(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def sqdist(a, b):
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.clip(d, 0, None)


# ------------------------------------------------------------------ distances between distributions
def frechet_fast(a, b):
    """Frechet distance. Tr sqrt(S1 S2) = sum of the singular values of Ac Bc^T / sqrt((n1-1)(n2-1))."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ac, bc = a - a.mean(0), b - b.mean(0)
    s = np.linalg.svd(ac @ bc.T, compute_uv=False).sum() / np.sqrt((len(a) - 1) * (len(b) - 1))
    t1, t2 = (ac ** 2).sum() / (len(a) - 1), (bc ** 2).sum() / (len(b) - 1)
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + t1 + t2 - 2 * s)


def kid_unbiased(a, b):
    """Unbiased MMD^2, kernel (x.y/d + 1)^3, with all the samples (no subsampling)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d, m, n = a.shape[1], len(a), len(b)
    kxx, kyy, kxy = ((x @ y.T / d + 1.0) ** 3 for x, y in ((a, a), (b, b), (a, b)))
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def cmmd(a, b, sigma=CMMD_SIGMA, scale=CMMD_SCALE, unbiased=False):
    """MMD^2 with the Gaussian kernel exp(-|x-y|^2 / (2 sigma^2)) on L2-normalized embeddings, times `scale`."""
    x, y = l2n(a), l2n(b)
    g = 1.0 / (2 * sigma ** 2)
    kxx, kyy, kxy = (np.exp(-g * sqdist(p, q)) for p, q in ((x, x), (y, y), (x, y)))
    if unbiased:
        m, n = len(x), len(y)
        return float(scale * ((kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1)) - 2 * kxy.mean()))
    return float(scale * (kxx.mean() + kyy.mean() - 2 * kxy.mean()))


# ------------------------------------------------------------------ fidelity and coverage (k neighbors)
def knn_radius(x, k=K_NN):
    """Euclidean distance of each point to its k-th nearest neighbor in the same set (not counting itself)."""
    d = sqdist(x, x)
    np.fill_diagonal(d, np.inf)
    k = min(k, len(x) - 1)
    return np.sqrt(np.partition(d, k - 1, axis=1)[:, k - 1])


def prdc(real, fake, k=K_NN):
    """Precision, recall (Kynkaanniemi 2019), density and coverage (Naeem 2020) with L2-normalized vectors."""
    r, f = l2n(real), l2n(fake)
    rr, rf = knn_radius(r, k), knn_radius(f, k)
    d = np.sqrt(sqdist(f, r))                                  # (m fake) x (n real)
    in_real = d <= rr[None, :]                                  # fake j inside the ball of real i
    d_rf = d.T                                                  # (n real) x (m fake)
    return {"precision": float(in_real.any(1).mean()),
            "recall": float((d_rf <= rf[None, :]).any(1).mean()),
            "density": float(in_real.sum() / (k * len(f))),
            "coverage": float(in_real.any(0).mean())}


def nn_to_real(real, fake):
    """Mean cosine distance of each synthetic item to its closest real one (0 = almost exact copy; shows near-copies)."""
    s = l2n(fake) @ l2n(real).T
    return float((1.0 - s.max(1)).mean())


def distribution_metrics(real, fake):
    """FD, KID, CMMD (biased and unbiased), precision/recall, density/coverage and synthetic->real NN."""
    out = {"fd": frechet_fast(real, fake), "kid": kid_unbiased(real, fake), "cmmd": cmmd(real, fake),
           "cmmd_unb": cmmd(real, fake, unbiased=True), "nn_real": nn_to_real(real, fake)}
    out.update(prdc(real, fake))
    return out


# ------------------------------------------------------------------ group-wise intervals
def _draw_groups(groups, rng, frac, replace):
    ug = np.array(sorted(set(groups)))
    pick = rng.choice(ug, max(2, int(round(len(ug) * frac))), replace=replace)
    return pick


def _rows(groups, pick):
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    return np.array([i for g in pick for i in by[g]], dtype=int)


def group_subsample_ci(fn, real, fake, g_real, g_fake, b=40, frac=0.8, seed=0, keys=None):
    """Variability by group SUBSAMPLING (without replacement, fraction `frac`) of both sets.

    It is not a classic bootstrap: with replacement the duplicates break the neighbor-based metrics (radius 0).
    Returns {metric: (p2.5, p97.5)} of the percentiles of the `b` replicates; it is a stability interval, not a
    confidence interval."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(b):
        ir = _rows(g_real, _draw_groups(g_real, rng, frac, False))
        jf = _rows(g_fake, _draw_groups(g_fake, rng, frac, False))
        if len(ir) > K_NN + 1 and len(jf) > K_NN + 1:
            reps.append(fn(real[ir], fake[jf]))
    keys = keys or list(reps[0])
    return {k: (float(np.percentile([r[k] for r in reps], 2.5)), float(np.percentile([r[k] for r in reps], 97.5)))
            for k in keys}


def auc_group_bootstrap(y, score, groups, b=300, seed=0):
    """95 % CI (percentiles) of the AUC by resampling groups with replacement on out-of-fold scores."""
    from sklearn.metrics import roc_auc_score
    rng, vals = np.random.default_rng(seed), []
    for _ in range(b):
        idx = _rows(groups, _draw_groups(groups, rng, 1.0, True))
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ grouped AUC
def grouped_auc(real, fake, g_real, g_fake, cls_real, cls_fake, seeds=(21, 22, 23)):
    """AUC real (0) vs synthetic (1) with StratifiedGroupKFold and 3 seeds, the same recipe as domain_gap.

    Returns the mean, the standard deviation across seeds, the group CI (bootstrap on the mean of the OOF scores) and
    the per-class AUC (inside each class, with the OOF scores of the global classifier)."""
    import domain_gap as dg
    from sklearn.metrics import roc_auc_score
    x, y = dg._xy(real, fake)
    grp = np.array(list(g_real) + list(g_fake))
    runs = [dg.cv_classify(x, y, seed=s, groups=grp) for s in seeds]
    oof = np.mean([r["oof"] for r in runs], axis=0)
    cls = np.array(list(cls_real) + list(cls_fake))
    per = {c: float(roc_auc_score(y[cls == c], oof[cls == c])) for c in sorted(set(cls))
           if len(set(y[cls == c])) == 2 and (cls == c).sum() >= 10}
    aucs = [r["auc"] for r in runs]
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_ci": auc_group_bootstrap(y, oof, grp), "per_class": per, "n_real": len(real), "n_fake": len(fake)}


# ------------------------------------------------------------------ TSTR
def tstr(train_x, train_y, test_x, test_y, classes, c=0.1, test_groups=None, b=300, seed=0):
    """Multiclass logistic regression (standardized, C = 0.1) trained on train and evaluated on test.

    Returns balanced accuracy, per-class recall and, if `test_groups` is given, the 95 % CI by bootstrap of test groups."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keep = [i for i, t in enumerate(train_y) if t in classes]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    clf.fit(np.asarray(train_x)[keep], np.asarray(train_y)[keep])
    tk = [i for i, t in enumerate(test_y) if t in classes]
    tx, ty = np.asarray(test_x)[tk], np.asarray(test_y)[tk]
    pred = clf.predict(tx)
    rec = {k: float((pred[ty == k] == k).mean()) for k in classes if (ty == k).any()}
    out = {"bacc": float(np.mean(list(rec.values()))), "recall": rec, "n_train": len(keep), "n_test": len(tk)}
    if test_groups is not None:
        out["bacc_ci"] = _bacc_bootstrap(pred, ty, [test_groups[i] for i in tk], b, seed)
    return out


def _bacc_bootstrap(pred, ty, groups, b, seed):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(b):
        idx = _rows(groups, _draw_groups(groups, rng, 1.0, True))
        r = [(pred[idx][ty[idx] == k] == k).mean() for k in sorted(set(ty[idx]))]
        vals.append(float(np.mean(r)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ validation against the AP (small n)
def spearman_min_p(n):
    """Exact two-sided minimum p of Spearman without ties with n observations: 2 / n!."""
    from math import factorial
    return min(1.0, 2.0 / factorial(n))


def pairwise_agreement(metric, ap, sense=1, min_ap_diff=0.0):
    """Fraction of batch pairs that the metric orders like the AP: (concordant, valid pairs).

    metric, ap: dict batch -> value. `sense` = +1 if higher = better. Pairs with |dAP| < min_ap_diff and pairs tied in
    the metric are ignored."""
    keys = sorted(set(metric) & set(ap))
    ok = tot = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            da, dm = ap[a] - ap[b], sense * (metric[a] - metric[b])
            if abs(da) < min_ap_diff or da == 0 or dm == 0:
                continue
            tot += 1
            ok += int((da > 0) == (dm > 0))
    return ok, tot


def spearman(metric, ap, sense=1):
    """Spearman between the (oriented: high = better) metric and the AP over the common batches; nan if n < 3."""
    from scipy.stats import spearmanr
    keys = sorted(set(metric) & set(ap))
    if len(keys) < 3:
        return float("nan")
    m, a = [sense * metric[k] for k in keys], [ap[k] for k in keys]
    if len(set(m)) < 2 or len(set(a)) < 2:
        return float("nan")
    return float(spearmanr(m, a)[0])


def rank_order(values, sense=1):
    """Names ordered from best to worst synthetic quality according to the oriented metric."""
    return [k for k, _ in sorted(values.items(), key=lambda kv: -sense * kv[1])]
