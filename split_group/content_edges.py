"""Content-similarity edges between images, computed from precomputed descriptors (CPU or GPU).

Descriptor files (``FEATURES_DIR``, written by ``split_group/features.py``), one row per image in the
order of the pooled annotations (ascending image id):
  dhash.npy       16x16 dHash of the 8 dihedral transforms, bit-packed (N x 8 x 32 uint8)
  embeddings.npy  ResNet-50 (ImageNet V2) embedding, mean over the 8 transforms, L2-normalised
                  (N x 2048 float16)

Criterion: min Hamming distance of the dHash <= TD (only if neither image is from CleanSea) or cosine
similarity >= TC; Roboflow-Roboflow pairs use the lower cosine threshold TC_RF. The original
orientation of the query image is compared with the 8 transforms of the other image, and a pair is
accepted if it meets the criterion in either direction.
"""
import numpy as np
import torch

TD, TC = 24, 0.95
DHASH_FILE, EMBEDDINGS_FILE = "dhash.npy", "embeddings.npy"


def features_exist(features_dir):
    return (features_dir / DHASH_FILE).exists() and (features_dir / EMBEDDINGS_FILE).exists()


def load_features(features_dir):
    """Return ``(dHash bits N x 8 x 256, embeddings N x 2048 float32)``."""
    hd = np.unpackbits(np.load(features_dir / DHASH_FILE), axis=2)[:, :, :256]
    em = np.load(features_dir / EMBEDDINGS_FILE).astype(np.float32)
    return hd, em


def _chunk(q_sign, s_all, e_q, e_all):
    """Minimum Hamming distance over transforms and cosine similarity for a block of queries."""
    dots = torch.einsum("qk,ntk->qnt", q_sign, s_all).amax(2)
    d = ((256 - dots) / 2).round()
    c = e_q @ e_all.T
    return d, c


def content_pairs(hd, em, is_cs, td=TD, tc=TC, device="cpu", block=256, cross_only=None, tc_rf=None):
    """Return ``(pairs, nn_dhash, nn_cosine)``.

    pairs: ``[(i, j, dhash, cosine)]`` with i < j that meet the criterion in either direction; for a
    pair found in both directions the smaller dHash distance is kept.
    tc_rf: cosine threshold for Roboflow-Roboflow pairs; ``None`` means ``tc``.
    cross_only: optional vector of partition labels; if given, only pairs with different labels are
    returned and ``nn_dhash``/``nn_cosine`` hold, per image, the closest neighbour in another
    partition (otherwise they stay at 999 / -1).
    """
    tc_rf = tc if tc_rf is None else tc_rf
    s_all = torch.from_numpy(hd.astype(np.float32) * 2 - 1).to(device)
    e_all = torch.from_numpy(em).to(device)
    cs = torch.from_numpy(np.asarray(is_cs, bool)).to(device)
    lab = None if cross_only is None else torch.from_numpy(np.asarray(cross_only)).to(device)
    n = len(hd)
    out, nn_d, nn_c = {}, np.full(n, 999.0), np.full(n, -1.0)
    with torch.no_grad():
        for a in range(0, n, block):
            b = min(a + block, n)
            d, c = _chunk(s_all[a:b, 0], s_all, e_all[a:b], e_all)
            both_rf = ~(cs[a:b, None] | cs[None, :])
            hit = ((d <= td) & both_rf) | (c >= tc) | ((c >= tc_rf) & both_rf)
            if lab is not None:
                other = lab[a:b, None] != lab[None, :]
                hit &= other
                dd = torch.where(other & both_rf, d, torch.full_like(d, 999.0))
                cc = torch.where(other, c, torch.full_like(c, -1.0))
                nn_d[a:b] = dd.amin(1).cpu().numpy()
                nn_c[a:b] = cc.amax(1).cpu().numpy()
            ii, jj = torch.nonzero(hit, as_tuple=True)
            for i, j in zip((ii + a).tolist(), jj.tolist()):
                if i != j:
                    key, dv = (min(i, j), max(i, j)), int(d[i - a, j])
                    if key not in out or dv < out[key][0]:
                        out[key] = (dv, round(float(c[i - a, j]), 6))
    pairs = [(i, j, dv, cv) for (i, j), (dv, cv) in sorted(out.items())]
    return pairs, nn_d, nn_c
