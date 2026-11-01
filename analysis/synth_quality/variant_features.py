"""Diagnosis of the copy-paste gap (1/2): features per region and low-level statistics (CPU only, TRAIN only).

Sets (all from TRAIN; never VAL or TEST):
  P   PASTED objects of the pilot j_v11_visible (origin = "pasted"), cropped from the composed image.
  RC  REAL objects kept in those same composed images (origin = "real"): control that goes through the same pipeline
      (same size, final JPEG q75) and only differs in not having been pasted.
  O   real TRAIN objects of the domain_gap sample (`samples.json`), cropped from the ORIGINAL file.
      `O_rc` = the same objects after recompressing the whole image to JPEG q75 (as the generator does): isolates the
      effect of double compression. `O` (not recompressed) reuses the domain_gap cache.
Region variants (all with the box enlarged by 20 % per side, as in domain_gap; gray fill = ImageNet mean):
  ctx    box with context (recipe of variant_pilot)
  inner  interior only: central rectangle (50 % of each side of the box), no borders or surroundings
  band   box border band only (between the box shrunk by 15 % and enlarged by 10 %)
  env    surroundings only: context without the box
The regions are geometric and identical for pasted and real objects (real objects have no mask; the generator does not
trace the rotation of the crop).

Inputs: the pilot folder (PILOT_J, default <PILOT_ROOT>/j_v11_visible) and the generator crop libraries (CROP_LIBRARY,
default RUNS_DIR/crop_library: `Extracted_objects_densea_v9train/` and `crops_poly/` with their manifests).

Usage (idle priority, 1 process, no GPU; requires `domain_gap.py --stage features` first):
  python analysis/synth_quality/variant_features.py
Output: RUNS_DIR/synth_quality/cache/variant_diag/<set>_<variant>.npy and <set>_stats.json (resumable per set).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variant_pilot_core as av  # noqa: E402

dg = av.dg
paths = av.paths
DCACHE = dg.CACHE / "variant_diag"
PILOT = Path(os.environ.get("PILOT_J", av.PILOT_ROOT / "j_v11_visible"))
CROP_LIBRARY = Path(os.environ.get("CROP_LIBRARY", paths.RUNS_DIR / "crop_library"))
LIB_OBJECTS, LIB_POLY = "Extracted_objects_densea_v9train", "crops_poly"
GRAY = (124, 116, 104)
VARIANTS = ("ctx", "inner", "band", "env")
DOMAINS = (("cleansea", "CleanSea"), ("neural_ocean", "Neural Ocean"), ("ocean_garbage", "Ocean Garbage"))
SOURCE_NAMES = {"CleanSea": "CleanSea", "Neural_Ocean": "Neural Ocean", "Ocean_garbage": "Ocean Garbage"}


def domain_of_path(p):
    """Domain (CleanSea / Neural Ocean / Ocean Garbage) according to the local path of a real image."""
    low = str(p).lower().replace("\\", "/")
    return next((name for key, name in DOMAINS if key in low), "?")


# ------------------------------------------------------------------ sets
def _cats(d):
    return {c["id"]: c["name"] for c in d["categories"]}


def _train_images():
    return json.load(open(paths.annotations_path("train"), encoding="utf8"))["images"]


def load_pilot():
    """Pasted (P) and kept real (RC) objects of the pilot, with their trace and the domain of the background."""
    d = json.load(open(PILOT / "synthetic_dataset.json", encoding="utf8"))
    names = _cats(d)
    bg_dom = {Path(i["file_name"]).name: SOURCE_NAMES.get(i["source"], "?") for i in _train_images()}
    imgs = {i["id"]: i for i in d["images"]}
    out = {"P": [], "RC": []}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        if a.get("iscrowd") or a["bbox"][2] < dg.MIN_BOX or a["bbox"][3] < dg.MIN_BOX:
            continue
        it = {"id": a["id"], "img": a["image_id"], "bbox": a["bbox"], "cls": names[a["category_id"]],
              "path": str(PILOT / "images" / im["file_name"]), "w": im["width"], "h": im["height"],
              "dest_dom": bg_dom.get(im["background"], "?"), "group": f"c{a['image_id']}"}
        if a.get("origin") == "pasted":
            it.update(scale=a.get("scale"), target_rel=a.get("target_rel_size"), src=a["source_object"])
            out["P"].append(it)
        else:
            out["RC"].append(it)
    return out


def load_reference():
    """Real TRAIN objects (domain_gap sample) with their domain and group (image)."""
    s = json.load(open(dg.WORK / "samples.json", encoding="utf8"))["real_train_obj"]
    return [dict(m, dest_dom=domain_of_path(m["path"]), group=m["path"], id=i) for i, m in enumerate(s)]


def crop_domains():
    """Domain of the source image of each crop (path -> domain), via the crop manifests and the TRAIN annotations."""
    rows = {i["id"]: SOURCE_NAMES.get(i["source"], "?") for i in _train_images()}
    out = {}
    m = json.load(open(CROP_LIBRARY / LIB_OBJECTS / "manifest_v9train.json"))["crops"]
    for c in m:
        out[f"{LIB_OBJECTS}/{c['cls']}/{c['ann']}.png"] = rows.get(c["image_id"], "?")
    p = json.load(open(CROP_LIBRARY / LIB_POLY / "manifest_poly.json"))["crops"]
    for c in p:
        out[f"{LIB_POLY}/{c['file']}"] = rows.get(c["image_id"], "?")
    return out


def source_key(src):
    """'/app/datasets/Extracted_objects_densea_v9train/Glove/1.png' -> 'Extracted_objects_densea_v9train/Glove/1.png'."""
    return "/".join(src.replace("\\", "/").split("/")[-3:])


# ------------------------------------------------------------------ regions
def regions(box, img_wh, ctx=dg.CTX):
    """Rectangles (image coordinates) -> dict: ctx, box B, inner."""
    x, y, bw, bh = box
    W, H = img_wh
    e = dg.expand_box(box, W, H, ctx)
    bx0, by0, bx1, by1 = max(0, int(round(x))), max(0, int(round(y))), min(W, int(round(x + bw))), min(H, int(round(y + bh)))
    sx, sy = 0.25 * (bx1 - bx0), 0.25 * (by1 - by0)
    inner = (int(bx0 + sx), int(by0 + sy), max(int(bx1 - sx), int(bx0 + sx) + 8), max(int(by1 - sy), int(by0 + sy) + 8))
    return {"ctx": e, "box": (bx0, by0, bx1, by1), "inner": inner}


def make_crops(arr, box):
    """PIL crops of the 4 region variants for a COCO box on the image `arr` (H,W,3 uint8)."""
    H, W = arr.shape[:2]
    r = regions(box, (W, H))
    x0, y0, x1, y1 = r["ctx"]
    ctx = arr[y0:y1, x0:x1].copy()
    bx0, by0, bx1, by1 = (v - o for v, o in zip(r["box"], (x0, y0, x0, y0)))
    h, w = ctx.shape[:2]
    bw, bh = bx1 - bx0, by1 - by0
    yy, xx = np.mgrid[0:h, 0:w]

    def inside(ex):   # box rectangle shifted by ex*side (ex<0 shrinks)
        return (xx >= bx0 - ex * bw) & (xx < bx1 + ex * bw) & (yy >= by0 - ex * bh) & (yy < by1 + ex * bh)

    gray = np.array(GRAY, np.uint8)
    band, env = ctx.copy(), ctx.copy()
    band[~(inside(0.10) & ~inside(-0.15))] = gray
    env[inside(0.0)] = gray
    ix0, iy0, ix1, iy1 = r["inner"]
    inner = arr[iy0:iy1, ix0:ix1]
    return {"ctx": Image.fromarray(ctx), "inner": Image.fromarray(inner), "band": Image.fromarray(band),
            "env": Image.fromarray(env)}, ctx, (bx0, by0, bx1, by1), (x0, y0), r["inner"]


# ------------------------------------------------------------------ low-level statistics
def _blockiness(g, x0, y0):
    """Ratio (jump at absolute 8x8 block boundaries) / (jump elsewhere); ~1 without a visible grid."""
    out = []
    for axis, off in ((1, x0), (0, y0)):
        d = np.abs(np.diff(g, axis=axis))
        idx = (np.arange(d.shape[axis]) + off + 1) % 8 == 0
        if idx.sum() == 0 or (~idx).sum() == 0:
            continue
        a, b = (d[:, idx], d[:, ~idx]) if axis == 1 else (d[idx], d[~idx])
        out.append(a.mean() / max(b.mean(), 1e-6))
    return float(np.mean(out)) if out else float("nan")


def _hf_ratio(g64):
    """Fraction of the spectral energy (without DC) above half the Nyquist frequency."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(g64 - g64.mean()))) ** 2
    yy, xx = np.mgrid[-32:32, -32:32]
    r = np.hypot(xx, yy)
    return float(f[r > 16].sum() / max(f.sum(), 1e-9))


def flat_noise(g):
    """Median standard deviation of the 20 % least textured 8x8 blocks (noise estimator in flat regions)."""
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    if h < 8 or w < 8:
        return float("nan")
    sd = g[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3)).ravel()
    return float(np.median(np.sort(sd)[: max(1, len(sd) // 5)]))


def low_level_stats(arr, box, img_wh):
    """Low-level statistics of an object (interior region) and of its geometry in the image."""
    crops, ctx, (bx0, by0, bx1, by1), (cx0, cy0), (ix0, iy0, ix1, iy1) = make_crops(arr, box)
    inner = arr[iy0:iy1, ix0:ix1]
    g = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g64 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA if min(g.shape) >= 64 else cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV).astype(np.float32)
    ang = hsv[..., 0] * 2 * np.pi / 180
    wgt = hsv[..., 1] / 255 + 1e-6
    lab_in = cv2.cvtColor(inner, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    frame = np.ones(ctx.shape[:2], bool)
    frame[by0:by1, bx0:bx1] = False
    lab_fr = cv2.cvtColor(ctx, cv2.COLOR_RGB2LAB)[frame].astype(np.float32) if frame.sum() > 20 else lab_in
    W, H = img_wh
    x, y, bw, bh = box
    return {"lapvar_native": float(cv2.Laplacian(g, cv2.CV_32F).var()),
            "lapvar64": float(cv2.Laplacian(g64, cv2.CV_32F).var()), "hf_ratio": _hf_ratio(g64),
            "noise_flat": flat_noise(g),
            "block": _blockiness(g, ix0, iy0), "sat": float(hsv[..., 1].mean()), "val": float(hsv[..., 2].mean()),
            "hue_cos": float((np.cos(ang) * wgt).sum() / wgt.sum()), "hue_sin": float((np.sin(ang) * wgt).sum() / wgt.sum()),
            "contrast_dE": float(np.linalg.norm(lab_in.mean(0) - lab_fr.mean(0))), "l_std": float(lab_in[:, 0].std()),
            "frame_rgb": [float(v) for v in ctx[frame].mean(0)] if frame.sum() > 20 else [float("nan")] * 3,
            "rel_size": float(np.sqrt(bw * bh / (W * H))), "log_aspect": float(np.log(bw / bh)),
            "cx": float((x + bw / 2) / W), "cy": float((y + bh / 2) / H),
            "border_dist": float(min(x, y, W - x - bw, H - y - bh) / max(W, H)), "min_side": float(min(bw, bh))}


# ------------------------------------------------------------------ extraction
def _load_arr(path, recompress):
    im = Image.open(path).convert("RGB")
    if recompress:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(im)


def _flush(model, tf, pend, feats):
    import torch
    if not pend:
        return
    with torch.no_grad():
        out = model(torch.stack([tf(im) for _, _, im in pend])).numpy().astype(np.float32)
    for (v, i, _), f in zip(pend, out):
        feats[v][i] = f
    pend.clear()


def process(name, items, recompress, variants=VARIANTS):
    """Features and statistics of a set (cached per set; resumable)."""
    DCACHE.mkdir(parents=True, exist_ok=True)
    if all((DCACHE / f"{name}_{v}.npy").exists() for v in variants) and (DCACHE / f"{name}_stats.json").exists():
        return
    from torchvision import transforms as T
    tf = T.Compose([T.Resize((dg.SIZE, dg.SIZE)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = dg.build_model()
    feats = {v: [None] * len(items) for v in variants}
    stats, pend, by = [None] * len(items), [], {}
    for i, it in enumerate(items):
        by.setdefault(it["path"], []).append(i)
    for n, (path, idxs) in enumerate(by.items()):
        arr = _load_arr(path, recompress)
        for i in idxs:
            crops = make_crops(arr, items[i]["bbox"])[0]
            pend += [(v, i, crops[v]) for v in variants]
            stats[i] = low_level_stats(arr, items[i]["bbox"], (arr.shape[1], arr.shape[0]))
        if len(pend) >= 64:
            _flush(model, tf, pend, feats)
        if n % 50 == 0:
            print(f"[{name}] image {n}/{len(by)}", flush=True)
    _flush(model, tf, pend, feats)
    for v in variants:
        np.save(DCACHE / f"{name}_{v}.npy", np.stack(feats[v]))
    json.dump(stats, open(DCACHE / f"{name}_stats.json", "w"))


def stats_only(name, items, recompress):
    """Recompute only the low-level statistics (no network) and overwrite <set>_stats.json."""
    by, stats = {}, [None] * len(items)
    for i, it in enumerate(items):
        by.setdefault(it["path"], []).append(i)
    for path, idxs in by.items():
        arr = _load_arr(path, recompress)
        for i in idxs:
            stats[i] = low_level_stats(arr, items[i]["bbox"], (arr.shape[1], arr.shape[0]))
    json.dump(stats, open(DCACHE / f"{name}_stats.json", "w"))
    print(f"[{name}] statistics recomputed", flush=True)


def main():
    dg.lower_priority()
    pil, ref = load_pilot(), load_reference()
    if "--stats-only" in sys.argv:
        for n, it, rc in (("P", pil["P"], False), ("RC", pil["RC"], False), ("Orc", ref, True), ("Oplain", ref, False)):
            stats_only(n, it, rc)
        return
    process("P", pil["P"], recompress=False)
    process("RC", pil["RC"], recompress=False, variants=("ctx",))
    process("Orc", ref, recompress=True)
    process("Oplain", ref, recompress=False, variants=("ctx",))
    print("done")


if __name__ == "__main__":
    main()
