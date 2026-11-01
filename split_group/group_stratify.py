"""Multi-label stratified partition BY GROUPS.

Group-level variant of iterative stratification (Sechidis et al., 2011): groups are assigned one at a
time, starting with those that contain the rarest classes, to the partition with the largest relative
deficit. The procedure runs with R random orders (seeded), the assignment with the lowest
cost is kept, and a local search then moves single groups whenever the cost decreases.

Stratified dimensions: boxes per class (8), images with each class (8), images per source (3) and total
images. The first 8 columns are always the boxes per class.
Hard constraint: every partition holds at least ``min_per_class`` boxes of each class.
"""
import numpy as np


def group_matrix(groups, img_classes, img_source, sources, n_classes):
    """G x D matrix: [boxes per class | images with the class | images per source | images]."""
    d = 2 * n_classes + len(sources) + 1
    m = np.zeros((len(groups), d), dtype=np.float64)
    for g, members in enumerate(groups):
        for i in members:
            for c in img_classes[i]:
                m[g, c] += 1
            for c in set(img_classes[i]):
                m[g, n_classes + c] += 1
            m[g, 2 * n_classes + sources.index(img_source[i])] += 1
            m[g, -1] += 1
    return m


def _cost(cur, target, n_classes, min_per_class):
    """Chi-square-like deviation from the target plus a hard penalty for missing boxes."""
    t = np.maximum(target, 1e-9)
    chi = ((cur - target) ** 2 / t).sum()
    short = np.maximum(min_per_class - cur[:, :n_classes], 0).sum()
    return chi + 1e4 * short


def _greedy(m, target, order):
    """Assign the groups in ``order`` to the partition with the largest weighted relative deficit."""
    k = target.shape[0]
    cur = np.zeros_like(target)
    assign = np.empty(len(m), dtype=np.int64)
    t = np.maximum(target, 1e-9)
    for g in order:
        need = (target - cur) / t                       # relative deficit per partition and dimension
        w = m[g] > 0
        score = (need[:, w] * m[g, w]).sum(1) / max(m[g, w].sum(), 1)
        s = int(np.argmax(score + 1e-9 * np.arange(k)[::-1]))
        assign[g] = s
        cur[s] += m[g]
    return assign, cur


def _order(m, n_classes, rng):
    """Groups with rare classes first; random order within the same rarity level."""
    totals = m[:, :n_classes].sum(0)
    rank = np.argsort(np.argsort(totals))               # 0 = rarest class
    has = m[:, :n_classes] > 0
    rarest = np.where(has.any(1), np.where(has, rank[None, :], 99).min(1), 99)
    perm = rng.permutation(len(m))
    return perm[np.lexsort((-m[perm, -1] * rng.uniform(0.5, 1.5, len(m)), rarest[perm]))]


def _local_search(m, target, assign, cur, n_classes, min_pc, rng, sweeps=4):
    """Move single groups between partitions while the cost decreases."""
    k = target.shape[0]
    best = _cost(cur, target, n_classes, min_pc)
    for _ in range(sweeps):
        improved = False
        for g in rng.permutation(len(m)):
            s0 = assign[g]
            for s1 in range(k):
                if s1 == s0:
                    continue
                cur[s0] -= m[g]; cur[s1] += m[g]
                c = _cost(cur, target, n_classes, min_pc)
                if c < best - 1e-12:
                    best, assign[g], s0, improved = c, s1, s1, True
                else:
                    cur[s1] -= m[g]; cur[s0] += m[g]
        if not improved:
            break
    return assign, cur, best


def split_groups(m, ratios, n_classes, seed=21, restarts=300, min_per_class=None):
    """Return ``(assignment per group, counts per partition, cost, chosen restart)``."""
    ratios = np.asarray(ratios, dtype=np.float64)
    target = ratios[:, None] * m.sum(0)[None, :]
    if min_per_class is None:
        min_per_class = np.zeros(n_classes)
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf, -1)
    for r in range(restarts):
        assign, cur = _greedy(m, target, _order(m, n_classes, rng))
        c = _cost(cur, target, n_classes, min_per_class)
        if c < best[2]:
            best = (assign.copy(), cur.copy(), c, r)
    assign, cur, c = _local_search(m, target, best[0], best[1], n_classes, min_per_class, rng)
    return assign, cur, c, best[3]
