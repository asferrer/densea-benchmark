"""Quality of the pasted crops: size, alpha coverage (rectangular?), border halo and color mismatch.

Each crop of the libraries (`Extracted_objects_densea_v9train`, `crops_poly`) is an RGBA PNG with binary alpha; the RGB
outside the alpha keeps the ORIGINAL background. From it we measure:
  crop_min     smaller side of the crop in pixels (before scaling).
  coverage     fraction of pixels with alpha > 0; `rect` = coverage > 0.95 (the crop drags its original background).
  halo         1 - d(border, old background) / d(interior, old background) in RGB: ~0 if the border (3 px inside the
               alpha) looks like the interior; ~1 if it looks like the original water around it (the border keeps the original surroundings).
  bg_mismatch  RGB distance between the mean color of the ORIGINAL background of the crop (3 px outside the alpha) and
               that of the surroundings in the target image (frame of the enlarged box).
"""
import cv2
import numpy as np
from PIL import Image

import variant_features as gf


def _mean(rgb, m):
    return rgb[m].mean(0) if m.any() else None


def one_crop(path):
    """Metrics of one RGBA crop PNG."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    rgb, alpha = a[..., :3].astype(np.float32), a[..., 3] > 127
    h, w = alpha.shape
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    a8 = alpha.astype(np.uint8)
    ring_in = alpha & ~cv2.erode(a8, k3).astype(bool)
    ring_out = cv2.dilate(a8, k3).astype(bool) & ~alpha
    interior = cv2.erode(a8, k5).astype(bool)
    if interior.sum() < 5:
        interior = cv2.erode(a8, k3).astype(bool)
    m_in, m_out, m_int = _mean(rgb, ring_in), _mean(rgb, ring_out), _mean(rgb, interior)
    halo = float("nan")
    if m_in is not None and m_out is not None and m_int is not None:
        den = np.linalg.norm(m_int - m_out)
        halo = float(np.clip(1 - np.linalg.norm(m_in - m_out) / max(den, 8.0), -1, 2))
    ys, xs = np.where(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy0, cy1, cx0, cx1 = (int(y0 + 0.25 * (y1 - y0)), int(y1 - 0.25 * (y1 - y0)), int(x0 + 0.25 * (x1 - x0)), int(x1 - 0.25 * (x1 - x0)))
    g = cv2.cvtColor(a[max(cy0, 0):max(cy1, cy0 + 8), max(cx0, 0):max(cx1, cx0 + 8), :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    g64 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA if min(g.shape) >= 64 else cv2.INTER_LINEAR)
    return {"src_hf": gf._hf_ratio(g64), "src_lapvar64": float(cv2.Laplacian(g64, cv2.CV_32F).var()),
            "crop_min": float(min(h, w)), "coverage": float(alpha.mean()), "rect": bool(alpha.mean() > 0.95),
            "halo": halo, "bg_rgb": None if m_out is None else [float(v) for v in m_out]}


def crop_quality(P, stats):
    """List (aligned with P) of crop metrics + mismatch with the target surroundings."""
    out = []
    cache = {}
    for p, s in zip(P, stats):
        key = gf.source_key(p["src"])
        if key not in cache:
            cache[key] = one_crop(gf.CROP_LIBRARY / key)
        q = dict(cache[key])
        fr, bg = np.array(s["frame_rgb"], float), q.pop("bg_rgb")
        q["bg_mismatch"] = float(np.linalg.norm(fr - np.array(bg))) if bg is not None and np.isfinite(fr).all() else float("nan")
        out.append(q)
    return out


def _sharp(rgba, alpha_thr=127):
    """(hf_ratio, lapvar64) of the central region (50 %) of the alpha box of an RGBA image."""
    ys, xs = np.where(rgba[..., 3] > alpha_thr)
    if len(ys) < 16:
        return float("nan"), float("nan")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy0, cy1 = int(y0 + 0.25 * (y1 - y0)), max(int(y1 - 0.25 * (y1 - y0)), int(y0 + 0.25 * (y1 - y0)) + 4)
    cx0, cx1 = int(x0 + 0.25 * (x1 - x0)), max(int(x1 - 0.25 * (x1 - x0)), int(x0 + 0.25 * (x1 - x0)) + 4)
    g = cv2.cvtColor(rgba[cy0:cy1, cx0:cx1, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    g64 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA if min(g.shape) >= 64 else cv2.INTER_LINEAR)
    return gf._hf_ratio(g64), float(cv2.Laplacian(g64, cv2.CV_32F).var())


def emulate_geometry(P, seed=21):
    """Emulate the deterministic generator stages on the crop (pre-blur, random rotation, recorded scale).

    Returns, per object, the sharpness after each stage: source, + pre-blur, + rotation, + scale. Replica of the
    generator's `_transform_object`: Gaussian pre-blur if min(side) < 100 (k = 3 if > 50, otherwise 5; sigma 0.5), rotation
    ±45° INTER_CUBIC and scale INTER_AREA (< 1) or LANCZOS4 (> 1). It does not include the blur budget.
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in P:
        a = np.asarray(Image.open(gf.CROP_LIBRARY / gf.source_key(p["src"])).convert("RGBA")).copy()
        st = [_sharp(a)]
        if min(a.shape[:2]) < 100:
            k = 3 if min(a.shape[:2]) > 50 else 5
            a[..., :3] = cv2.GaussianBlur(a[..., :3], (k, k), 0.5)
        st.append(_sharp(a))
        h, w = a.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-45, 45)), 1.0)
        nw, nh = int(h * abs(M[0, 1]) + w * abs(M[0, 0])), int(h * abs(M[0, 0]) + w * abs(M[0, 1]))
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2
        a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        st.append(_sharp(a))
        s = p["scale"]
        a = cv2.resize(a, (max(int(nw * s), 4), max(int(nh * s), 4)), interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
        st.append(_sharp(a))
        out.append(st)
    return np.array(out)   # (n, 4 stages, 2 metrics)
