"""Groups of images that come from the same original photograph or scene.

A group is a connected component of the graph whose edges are:
  (a) same normalised name key (Roboflow sources: without ``_jpg.rf.<hash>``, without a residual
      ``_jpeg/_png`` tag, without ``__fliph/__flipv`` and without ``(aN_)+`` prefixes; CleanSea: same
      file name);
  (b) content similarity: 16x16 dHash under the 8 dihedral transforms (Hamming <= 24/256) or cosine of
      ResNet-50 embeddings (8-flip average) >= 0.95; for pairs with a CleanSea image only the cosine
      criterion applies, because dHash gives false positives on homogeneous sea floors;
  (c) CleanSea: same dive (all of its cameras) according to the file name; optionally only same dive
      and camera (one video sequence).
"""
import collections
import os
import re

RF = re.compile(r"_(jpg|jpeg|png|JPG|JPEG|PNG)\.rf\.[0-9a-f]{32}$")
EXT = re.compile(r"_(jpg|jpeg|png|JPG|JPEG|PNG)$")
FLIP = re.compile(r"__flip[hv]")
APFX = re.compile(r"^(a\d+_)+")
# Dive (HPD/KAIKO/2K/3K/6K + 4 digits), camera and frame number. From dive HPD1938 on, two cameras
# (HDTV1/HDTV2) record at once with the same frame numbers (for example HPD2042HDTV10014 and
# HPD2042HDTV20014), so the default mode groups by the whole dive, which also covers dive + camera.
SEQ = re.compile(r"^((?:HPD|KAIKO|\dK)\d{4})(HDTV\d|HDTV|OUT|INa|IN|ASC|OLD)(\d{4})")


def name_key(file_name, source):
    """Normalised name key (criterion a); only the base name of ``file_name`` is used."""
    base = os.path.basename(file_name)
    if source == "CleanSea":
        return "CS:" + base
    stem = os.path.splitext(base)[0]
    k = EXT.sub("", RF.sub("", stem))
    k = re.sub(r"_+$", "", FLIP.sub("", k))
    return "RF:" + APFX.sub("", k)


def cleansea_sequence(file_name, mode="dive"):
    """CleanSea video sequence (criterion c). ``mode``: ``dive`` (default) or ``dive_camera``.

    The two files without the dive pattern (P1150084 and P1150097, camera photos) form one sequence.
    """
    m = SEQ.match(os.path.basename(file_name))
    if not m:
        return "CSSEQ:P115"
    if mode == "dive":
        return "CSSEQ:" + m.group(1)
    return f"CSSEQ:{m.group(1)}|{m.group(2)}"


class UnionFind:
    """Union-find whose root is always the smallest index of its component."""

    def __init__(self, n):
        self.par = list(range(n))

    def find(self, x):
        while self.par[x] != x:
            self.par[x] = self.par[self.par[x]]
            x = self.par[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.par[max(ra, rb)] = min(ra, rb)


def _union_by_key(uf, keys):
    first = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in first:
            uf.union(i, first[k])
        else:
            first[k] = i


def build_groups(rows, pairs, cs_mode="dive"):
    """Return ``(groups, stats, name_keys, sequences)``.

    rows: ``[{'file_name', 'source', ...}]`` in the row order of the descriptor arrays.
    pairs: ``[(i, j, dhash, cosine)]`` content edges already filtered by threshold.
    groups: lists of row indices, ordered by their first row.
    """
    n = len(rows)
    uf = UnionFind(n)
    nkeys = [name_key(r["file_name"], r["source"]) for r in rows]
    _union_by_key(uf, nkeys)
    n_after_name = len({uf.find(i) for i in range(n)})
    for i, j, _d, _c in pairs:
        uf.union(int(i), int(j))
    n_after_content = len({uf.find(i) for i in range(n)})
    seqs = [cleansea_sequence(r["file_name"], cs_mode) if r["source"] == "CleanSea" else None
            for r in rows]
    _union_by_key(uf, seqs)
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    groups = sorted(comp.values(), key=lambda g: g[0])
    stats = {"images": n, "content_edges": len(pairs), "components_name": n_after_name,
             "components_name_content": n_after_content, "groups_final": len(groups),
             "largest_group": max(len(g) for g in groups), "cleansea_mode": cs_mode}
    return groups, stats, nkeys, seqs
