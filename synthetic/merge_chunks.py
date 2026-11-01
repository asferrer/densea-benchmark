"""Merge the chunks of a copy-paste batch into one COCO file and copy the images into one folder.

Input layout (written by the generator's tools/run_variants.py, one call per chunk):
  <src>/chunk_<k>/<variant>/{images/, synthetic_dataset.json, variant.json}      k = 1..N
Output layout (a NEW folder; an existing one is never overwritten):
  <dst>/chunk_<k>/images/*.jpg   and   <dst>/synthetic_dataset.json
`file_name` in the merged file is ``chunk_<k>/images/<file>`` (relative to <dst>); image and annotation ids are
renumbered without collisions and every image gets a `chunk` field. <dst> is a batch folder under
``DATA_ROOT/synthetic`` (``copy_paste``, ``copy_paste_r25``, ``copy_paste_r50``).

A summary JSON is written (images, pasted and real boxes per class against the per-class targets of the
request, rejected images and hours per chunk).

Usage:
  python synthetic/merge_chunks.py --src <chunks folder> --dst <new batch folder> --chunks 6 \
      --request <batch_request.json> --name copy_paste [--variant s_drop] [--summary FILE] [--dry-run]
"""
import argparse
import collections
import json
import shutil
import sys
from pathlib import Path


def load_chunk(src, name, variant):
    d = Path(src) / name / variant
    with open(d / "synthetic_dataset.json", encoding="utf8") as f:
        coco = json.load(f)
    with open(d / "variant.json", encoding="utf8") as f:
        info = json.load(f)
    return coco, info


def merge(chunks, batch_name):
    """Single COCO from ``[(chunk name, dataset)]``; every chunk must share the category list."""
    cats = chunks[0][1]["categories"]
    out = {"info": {"description": f"Synthetic copy-paste batch {batch_name}, {len(chunks)} chunks"},
           "licenses": [], "categories": cats, "images": [], "annotations": []}
    next_img, next_ann = 0, 0
    for name, ds in chunks:
        if ds["categories"] != cats:
            sys.exit(f"different categories in {name}")
        remap = {}
        for im in ds["images"]:
            remap[im["id"]] = next_img
            out["images"].append(dict(im, id=next_img, file_name=f"{name}/images/{im['file_name']}", chunk=name))
            next_img += 1
        for a in ds["annotations"]:
            out["annotations"].append(dict(a, id=next_ann, image_id=remap[a["image_id"]]))
            next_ann += 1
    return out


def summary(ds, variants, targets):
    names = {c["id"]: c["name"] for c in ds["categories"]}
    by = collections.defaultdict(collections.Counter)
    for a in ds["annotations"]:
        by[a.get("origin", "pasted")][names[a["category_id"]]] += 1
    return {"images": len(ds["images"]), "annotations": len(ds["annotations"]),
            "pasted_total": sum(by["pasted"].values()), "real_total": sum(by["real"].values()),
            "pasted_per_class": dict(by["pasted"]), "real_per_class": dict(by["real"]),
            "target_per_class": targets,
            "rejected_per_chunk": {n: v["rejected"] for n, v in variants.items()},
            "hours_per_chunk": {n: round(v["seconds"] / 3600, 2) for n, v in variants.items()}}


def copy_images(src, dst, names, variant):
    for name in names:
        out = Path(dst) / name / "images"
        out.mkdir(parents=True, exist_ok=False)
        for f in (Path(src) / name / variant / "images").glob("*.jpg"):
            shutil.copy2(f, out / f.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="folder that contains chunk_1 ... chunk_N")
    ap.add_argument("--dst", required=True, help="NEW batch folder")
    ap.add_argument("--chunks", type=int, required=True, help="number of chunks")
    ap.add_argument("--request", required=True, help="job request JSON used for generation (per-class targets)")
    ap.add_argument("--name", required=True, help="batch name, written in info.description")
    ap.add_argument("--variant", default="s_drop", help="variant folder inside each chunk")
    ap.add_argument("--summary", help="summary JSON path (default: <dst>/summary.json)")
    ap.add_argument("--dry-run", action="store_true", help="print the summary without copying")
    a = ap.parse_args()
    names = [f"chunk_{k}" for k in range(1, a.chunks + 1)]
    loaded = {n: load_chunk(a.src, n, a.variant) for n in names}
    ds = merge([(n, loaded[n][0]) for n in names], a.name)
    targets = json.load(open(a.request, encoding="utf8"))["targets_per_class"]
    rep = summary(ds, {n: loaded[n][1] for n in names}, targets)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.dry_run:
        return
    dst = Path(a.dst)
    if dst.exists():
        sys.exit(f"{dst} already exists: it is not overwritten")
    copy_images(a.src, dst, names, a.variant)
    json.dump(ds, open(dst / "synthetic_dataset.json", "w", encoding="utf8"))
    n = sum(1 for _ in dst.glob("chunk_*/images/*.jpg"))
    if n != rep["images"]:
        sys.exit(f"copied {n} images but the COCO lists {rep['images']}")
    json.dump(rep, open(a.summary or dst / "summary.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("copied", n, "images to", dst)


if __name__ == "__main__":
    main()
