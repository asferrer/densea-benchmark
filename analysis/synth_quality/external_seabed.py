"""Gap of the external "Synthetic Seabed" set (Zenodo 10.5281/zenodo.21208479, CC BY-NC 4.0) with the kit's metrics.

External pipeline: SDXL (backgrounds) + FLUX.1 Fill (object) + SAM (mask/box) + water filter. It is measured with the
same kit as `variant_pilot`: object crop with context (`domain_gap.expand_box`, CTX 0.20), ResNet-50, per-class
quotas, real TRAIN items from the domain_gap cache and grouped CV with 3 seeds (one group per synthetic image: there
is no reused-background identifier, every image has its own background). CPU only, low priority, never TEST.

The set is treated as untrusted data: only PNG, TXT and JSONL files are read with PIL/numpy/json; nothing is executed.
Expected layout of the extracted set (EXT_ROOT, default DATA_ROOT/external/synthetic_seabed): final/, labels/,
labels_seg/, inserted/, backgrounds/ and manifest.jsonl. Requires `variant_pilot.py --stage features` first (it reads
its plan.json and the m_pos features).

Usage (a single process):
  python -I analysis/synth_quality/external_seabed.py --stage features
  python -I analysis/synth_quality/external_seabed.py --stage analyze
Output: out/external_seabed.json; mosaic and caches in RUNS_DIR/synth_quality/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variant_measures as measures  # noqa: E402
import variant_pilot_core as av  # noqa: E402

dg = av.dg
OUT = dg.OUT
ECACHE = dg.CACHE / "external_seabed"
EXT_ROOT = Path(os.environ.get("EXT_ROOT", dg.DATA_ROOT / "external" / "synthetic_seabed"))
NAME = "external_seabed"
N_IMG = 300

# YOLO class ids of the set (order of the Zenodo record; checked by crossing labels/ with the manifest).
EXT_CLASSES = ["Plastic_Bag", "Can", "Packaging_Bag", "Bottle", "Fishing_Net", "Plastic_Debris", "Wood", "Car_Bumper",
               "Towel", "Pipe", "Rope", "Glove", "Basket", "Squared_Can", "Metal_Debris", "Tire", "Shoe",
               "WashingMachine", "Metal_Chain"]
# Equivalence with our 8 classes; anything else is dropped. Squared_Can (square can/drum) and Metal_Chain are left out on
# purpose: they are not "Can" or "Metal_Debris" in our labeling. Mask has no equivalent.
CLASS_MAP = {"Bottle": "Bottle", "Can": "Can", "Fishing_Net": "Fishing_Net", "Glove": "Glove",
             "Metal_Debris": "Metal_Debris", "Plastic_Debris": "Plastic_Debris", "Tire": "Tire"}


def map_class(ext_name):
    """Our class equivalent to a class of the external set, or None if it is dropped."""
    return CLASS_MAP.get(ext_name)


def yolo_to_coco(parts, w, h):
    """Normalized YOLO line [id, xc, yc, bw, bh] -> (id, [x, y, w, h]) in pixels."""
    i, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
    return i, [round((xc - bw / 2) * w), round((yc - bh / 2) * h), round(bw * w), round(bh * h)]


def poly_area_norm(coords):
    """Normalized area (shoelace formula) of the YOLO-seg polygon [x1, y1, x2, y2, ...]."""
    p = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def load_external(root=EXT_ROOT, size=768):
    """Final images and annotations (mapped classes only) in the `domain_gap` format. Ids = image number."""
    imgs, anns = {}, []
    for f in sorted((Path(root) / "final").glob("img_*.png")):
        iid = int(f.stem.split("_")[1])
        imgs[iid] = {"id": iid, "path": str(f), "source": "synthetic", "w": size, "h": size}
        lab = Path(root) / "labels" / f"{f.stem}.txt"
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            ci, box = yolo_to_coco(line.split(), size, size)
            cls = map_class(EXT_CLASSES[ci])
            if cls:
                anns.append({"img": iid, "bbox": box, "cls": cls})
    return imgs, anns


def external_quotas():
    """Quotas of variant_pilot (plan.json) capped by what is available: Mask = 0 (no equivalent)."""
    plan = json.load(open(av.ACACHE / "plan.json"))
    _, anns = load_external()
    have = {c: sum(a["cls"] == c for a in anns if a["bbox"][2] >= dg.MIN_BOX and a["bbox"][3] >= dg.MIN_BOX)
            for c in dg.CLASSES}
    return {c: min(plan["quotas"][c], have[c]) for c in dg.CLASSES}, have, plan["quotas"]


def stage_features():
    """Extract and cache the features (ResNet-50, CPU) of 300 images and of the objects within the quotas."""
    quotas, _, _ = external_quotas()
    imgs, anns = load_external()
    items = sorted(imgs.values(), key=lambda i: i["path"])[:N_IMG]
    samp = {"img": items, "obj": dg.sample_objects(imgs, anns, quotas)[0]}
    ECACHE.mkdir(parents=True, exist_ok=True)
    model = dg.build_model()
    for kind in ("img", "obj"):
        np.save(ECACHE / f"{NAME}_{kind}.npy", dg.extract(f"{NAME}/{kind}", samp[kind], kind, model))
    json.dump(samp, open(ECACHE / f"{NAME}_meta.json", "w", encoding="utf8"))
    json.dump({"n_img": N_IMG, "quotas": quotas}, open(ECACHE / "plan.json", "w"))


def drop_class(v, cls):
    """Copy of a variant without the object rows of one class (to compare with the same set of classes)."""
    keep = [i for i, m in enumerate(v["obj_meta"]) if m["cls"] != cls]
    out = dict(v, obj=v["obj"][keep], obj_meta=[v["obj_meta"][i] for i in keep])
    if "obj_g" in v:                    # per-row groups (attach_groups): filtered like the rows
        out["obj_g"] = [v["obj_g"][i] for i in keep]
    return out


def margin_stats(root=EXT_ROOT, size=768, max_images=None):
    """Fit of the external boxes: box vs SAM polygon, polygon fill and actual pixel change.

    `fill` = polygon area / box area; `dev_poly` = |box width+height - those of the polygon bbox| (px);
    `change_iou` = IoU between the box and the bbox of the pixels that FLUX changed with respect to the background
    (inserted vs backgrounds, threshold 12/255 on the channel maximum), restricted to the insertion region
    (`placed.box`) of the manifest."""
    from PIL import Image
    man = [json.loads(line) for line in open(Path(root) / "manifest.jsonl", encoding="utf8")][:max_images]
    rows = []
    for d in man:
        stem = d["image"][:-4]
        boxes = [l.split() for l in (Path(root) / "labels" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        segs = [l.split() for l in (Path(root) / "labels_seg" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        ins = np.asarray(Image.open(Path(root) / "inserted" / f"{stem}.png").convert("RGB"), dtype=np.int16)
        bg = np.asarray(Image.open(Path(root) / "backgrounds" / f"{stem}.png").convert("RGB"), dtype=np.int16)
        ch = np.abs(ins - bg).max(axis=2) > 12
        for b, s, p in zip(boxes, segs, d["placed"]):
            cls = map_class(p["class"])
            if cls:
                rows.append(_margin_row(cls, b, s, p, ch, size))
    return rows


def _margin_row(cls, b, s, placed, changed, size):
    _, (x, y, w, h) = yolo_to_coco(b, size, size)
    poly = np.asarray(s[1:], dtype=np.float64).reshape(-1, 2) * size
    pw, ph = np.ptp(poly[:, 0]), np.ptp(poly[:, 1])
    x0, y0, x1, y1 = placed["box"]
    sub = changed[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    iou = None
    if len(xs):
        cx0, cy0, cx1, cy1 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
        iw, ih = max(0, min(x + w, cx1) - max(x, cx0)), max(0, min(y + h, cy1) - max(y, cy0))
        iou = iw * ih / max(1, w * h + (cx1 - cx0) * (cy1 - cy0) - iw * ih)
    return {"cls": cls, "fill": poly_area_norm(s[1:]) / max(1e-9, float(b[3]) * float(b[4])),
            "dev_poly": abs(w - pw) + abs(h - ph), "box_vs_region": (w * h) / max(1, (x1 - x0) * (y1 - y0)),
            "change_iou": iou}


def margin_summary(rows):
    """Medians (and p10-p90) of each box-fit metric."""
    out = {"n": len(rows)}
    for k in ("fill", "dev_poly", "box_vs_region", "change_iou"):
        v = np.array([r[k] for r in rows if r[k] is not None], dtype=np.float64)
        out[k] = {q: round(float(np.percentile(v, p)), 3) for q, p in (("p10", 10), ("p50", 50), ("p90", 90))}
    return out


def write_coco(root=EXT_ROOT, out_dir=None):
    """COCO of the mapped classes (absolute paths) so that `variant_measures` can be reused unchanged."""
    out_dir = Path(out_dir or dg.WORK / "external_seabed_coco" / "mapped")
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs, anns = load_external(root)
    cid = {c: i for i, c in enumerate(dg.CLASSES)}
    d = {"images": [{"id": i["id"], "file_name": i["path"], "width": i["w"], "height": i["h"]} for i in imgs.values()],
         "categories": [{"id": i, "name": c} for c, i in cid.items()],
         "annotations": [{"id": k + 1, "image_id": a["img"], "category_id": cid[a["cls"]], "bbox": a["bbox"],
                          "iscrowd": 0} for k, a in enumerate(anns)]}
    json.dump(d, open(out_dir / "synthetic_dataset.json", "w", encoding="utf8"))
    return out_dir.parent, out_dir.name


def stage_analyze(reference=None):
    quotas, have, plan_q = external_quotas()
    bg_map = av.background_group_map()
    ext = av.attach_groups(av.variant_data(NAME, ECACHE), NAME, bg_map)
    mp = drop_class(av.attach_groups(av.variant_data("m_pos"), "m_pos", bg_map), "Mask")
    real = av.real_subsets(N_IMG, quotas)
    rep = av.analyze_variants({NAME: ext, "m_pos_7classes": mp}, real, quotas)
    root, name = write_coco()
    rep["_meta"] = {"quotas": quotas, "available": have, "plan_v11": plan_q,
                    "margin": margin_summary(margin_stats()),
                    "measures": measures.measure_all(root, [name], reference=reference)}
    (dg.WORK / "mosaics").mkdir(parents=True, exist_ok=True)
    av.make_mosaic(NAME, dg.WORK / "mosaics" / "external_seabed.jpg", cache=ECACHE)
    json.dump(rep, open(OUT / "external_seabed.json", "w", encoding="utf8"), indent=1, default=float)
    print(json.dumps({k: rep[k]["object"] for k in (NAME, "m_pos_7classes")}, indent=1, default=float)[:1500])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["features", "analyze", "all"], default="all")
    ap.add_argument("--train-reference", default=str(measures.DEFAULT_REFERENCE))
    a = ap.parse_args()
    dg.lower_priority()
    if a.stage in ("features", "all"):
        stage_features()
    if a.stage in ("analyze", "all"):
        stage_analyze(a.train_reference)


if __name__ == "__main__":
    main()
