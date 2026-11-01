"""Direct measurements of a pilot batch to check each generator improvement without training anything (CPU, TRAIN only).

On the COCO and images of a variant (`synthetic_dataset.json`, `images/`) it measures, for the PASTED objects
(`origin = "pasted"` or no `origin`) and, as a control that goes through the same pipeline, for the KEPT REAL objects
(`origin = "real"`, only with copy-paste):

  * Sharpness in the central 50 % of the box at 64 x 64: `hf_ratio` (fraction of the energy above half the Nyquist
    frequency) and `lapvar64` (Laplacian variance).
  * Noise in flat regions (`noise_flat`, 8 x 8 blocks, native resolution).
  * Fraction of boxes touching the image border (within 3 px of a side).
  * Distribution of the pasting scale (COCO `scale`, needs the generator trace): quantiles and fractions < 0.5, in
    [0.6, 1.5] and > 1.5.

The reference for real TRAIN objects (`train_reference.json`: n, median hf_ratio/lapvar64/noise_flat, fraction of
boxes at the border) holds the statistics of the real TRAIN objects.

Usage:
  python analysis/synth_quality/variant_measures.py --root runs/pilots --variants j_v11_visible k_sharp
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "out" / "train_reference.json"
TOUCH_PX = 3
MIN_SIDE = 16
MEASURE_SIDE = 64
CENTRAL_FRACTION = 0.5     # the statistic is measured in the central 50 % of the box
FLAT_BLOCK = 8
FLAT_FRACTION = 0.2        # fraction of the least textured blocks used to estimate the noise
MIN_MEASURE_PIXELS = 8
SCALE_BANDS = (("< 0.5", 0.0, 0.5), ("0.5 - 0.6", 0.5, 0.6), ("0.6 - 1.5", 0.6, 1.5), ("> 1.5", 1.5, 1e9))


# ------------------------------------------------------------------ measures
def hf_ratio(g64):
    """Fraction of the spectral energy (without DC) above half the Nyquist frequency, in a 64 x 64 image."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(g64 - g64.mean()))) ** 2
    yy, xx = np.mgrid[-32:32, -32:32]
    return float(f[np.hypot(xx, yy) > 16].sum() / max(f.sum(), 1e-9))


def to_measure_size(gray):
    """Gray region resampled to 64 x 64 (AREA if larger, LINEAR otherwise)."""
    interp = cv2.INTER_AREA if min(gray.shape) >= MEASURE_SIDE else cv2.INTER_LINEAR
    return cv2.resize(gray, (MEASURE_SIDE, MEASURE_SIDE), interpolation=interp)


def central_region(gray, box):
    """Central 50 % of the box (x1, y1, x2, y2) of a gray image; None if it is too small."""
    x1, y1, x2, y2 = box
    m = (1.0 - CENTRAL_FRACTION) / 2.0
    cx1, cx2 = int(x1 + m * (x2 - x1)), int(x2 - m * (x2 - x1))
    cy1, cy2 = int(y1 + m * (y2 - y1)), int(y2 - m * (y2 - y1))
    cx2, cy2 = max(cx2, cx1 + 4), max(cy2, cy1 + 4)
    if cx2 - cx1 < MIN_MEASURE_PIXELS or cy2 - cy1 < MIN_MEASURE_PIXELS:
        return None
    return gray[cy1:cy2, cx1:cx2]


def flat_noise(gray):
    """Median standard deviation of the 20 % least textured 8 x 8 blocks."""
    h, w = (gray.shape[0] // FLAT_BLOCK) * FLAT_BLOCK, (gray.shape[1] // FLAT_BLOCK) * FLAT_BLOCK
    if h < FLAT_BLOCK or w < FLAT_BLOCK:
        return None
    shape = (h // FLAT_BLOCK, FLAT_BLOCK, w // FLAT_BLOCK, FLAT_BLOCK)
    sd = np.sort(gray[:h, :w].reshape(shape).std(axis=(1, 3)).ravel())
    return float(np.median(sd[: max(1, int(len(sd) * FLAT_FRACTION))])) if len(sd) >= 5 else None


def touches_border(box_xyxy, bg_wh, tol=TOUCH_PX):
    """True if the box (x1, y1, x2, y2) is within `tol` px of any side of the image."""
    return (box_xyxy[0] <= tol or box_xyxy[1] <= tol or bg_wh[0] - box_xyxy[2] <= tol
            or bg_wh[1] - box_xyxy[3] <= tol)


def _box_stats(gray, box):
    """hf_ratio, lapvar64 and noise_flat of the central 50 % of a COCO box (x, y, w, h); None if it cannot be measured."""
    h, w = gray.shape
    x, y, bw, bh = box
    x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if min(x2 - x1, y2 - y1) < MIN_SIDE:
        return None
    region = central_region(gray, (x1, y1, x2, y2))
    if region is None:
        return None
    g64 = to_measure_size(region)
    return {"hf_ratio": hf_ratio(g64), "lapvar64": float(cv2.Laplacian(g64, cv2.CV_32F).var()),
            "noise_flat": flat_noise(region), "touch": touches_border((x1, y1, x2, y2), (w, h))}


def _median(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return round(float(np.median(vals)), 5) if vals else None


def _group_summary(rows):
    return {"n": len(rows), "hf_ratio": _median(rows, "hf_ratio"), "lapvar64": _median(rows, "lapvar64"),
            "noise_flat": _median(rows, "noise_flat"),
            "border": round(float(np.mean([r["touch"] for r in rows])), 4) if rows else None}


def scale_summary(scales):
    """Quantiles of the scale and fraction of objects in each band."""
    if not len(scales):
        return {}
    arr = np.asarray(scales, dtype=np.float64)
    out = {"n": int(arr.size), **{f"p{q}": round(float(np.percentile(arr, q)), 3) for q in (10, 25, 50, 75, 90)}}
    out["bands"] = {name: round(float(((arr >= lo) & (arr < hi)).mean()), 3) for name, lo, hi in SCALE_BANDS}
    return out


def measure_variant(vdir, max_images=None):
    """{pasted, kept_real, scale} of a variant; each image is read once."""
    d = json.load(open(Path(vdir) / "synthetic_dataset.json", encoding="utf8"))
    imgs = {i["id"]: i for i in d["images"]}
    by_img = {}
    for a in d["annotations"]:
        if not a.get("iscrowd", 0):
            by_img.setdefault(a["image_id"], []).append(a)
    rows = {"pasted": [], "real": []}
    scales = []
    for k, (iid, anns) in enumerate(by_img.items()):
        if max_images and k >= max_images:
            break
        img = cv2.imread(str(Path(vdir) / "images" / imgs[iid]["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gray = img.astype(np.float32)
        for a in anns:
            m = _box_stats(gray, a["bbox"])
            if m is None:
                continue
            kind = "real" if a.get("origin") == "real" else "pasted"
            rows[kind].append(m)
            if kind == "pasted" and a.get("scale") is not None:
                scales.append(a["scale"])
    return {"pasted": _group_summary(rows["pasted"]), "kept_real": _group_summary(rows["real"]),
            "scale": scale_summary(scales)}


def train_reference(path=None):
    """Reference of the real TRAIN objects: n, median hf_ratio/lapvar64/noise_flat and border fraction."""
    return json.load(open(path or DEFAULT_REFERENCE, encoding="utf8"))


def measure_all(root, names, max_images=None, reference=None):
    out = {"_train_reference": train_reference(reference)}
    for n in names:
        if (Path(root) / n / "synthetic_dataset.json").exists():
            out[n] = measure_variant(Path(root) / n, max_images)
    return out


# ------------------------------------------------------------------ report
def _fmt(v):
    return "-" if v is None else str(v)


def md_lines(meas):
    """Sharpness/noise/border and scale tables for the pilot report."""
    ref = meas["_train_reference"]
    L = ["", "## Sharpness, noise, border and scale of the pasted objects (check of the generator improvements)", "",
         "Medians in the central 50 % of the box at 64 x 64 (hf_ratio, lapvar64) and at native resolution "
         "(noise_flat). `border` = fraction of boxes within 3 px of the image border. Reference: real TRAIN objects "
         f"(n = {ref['n']}); the \"kept real\" objects are the real ones of the same composed images (same pipeline, "
         "final JPEG included).", "",
         "| Variant | Group | n | hf_ratio | lapvar64 | noise_flat | border |", "|---|---|---|---|---|---|---|",
         f"| TRAIN real | real | {ref['n']} | {ref['hf_ratio']} | {ref['lapvar64']} | {ref['noise_flat']} | "
         f"{ref['border']} |"]
    for name, m in meas.items():
        if name.startswith("_"):
            continue
        for grp, label in (("pasted", "pasted"), ("kept_real", "kept real")):
            g = m[grp]
            if g["n"]:
                L.append(f"| {name} | {label} | {g['n']} | {_fmt(g['hf_ratio'])} | {_fmt(g['lapvar64'])} | "
                         f"{_fmt(g['noise_flat'])} | {_fmt(g['border'])} |")
    bands = [b[0] for b in SCALE_BANDS]
    L += ["", "Pasting scale (COCO `scale`; target of `crop_scale_range`: 0.6 - 1.5):", "",
          "| Variant | n | p10 | p25 | p50 | p75 | p90 | " + " | ".join(f"% {b}" for b in bands) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(bands)]
    for name, m in meas.items():
        s = m.get("scale") if not name.startswith("_") else None
        if s:
            L.append(f"| {name} | {s['n']} | {s['p10']} | {s['p25']} | {s['p50']} | {s['p75']} | {s['p90']} | "
                     + " | ".join(f"{s['bands'][b]:.0%}" for b in bands) + " |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--train-reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--out", help="output JSON (optional)")
    a = ap.parse_args()
    meas = measure_all(a.root, a.variants, a.max_images, a.train_reference)
    print("\n".join(md_lines(meas)))
    if a.out:
        json.dump(meas, open(a.out, "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    main()
