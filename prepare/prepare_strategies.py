"""Build the training JSON of each experiment from the real data and one synthetic batch.

Every strategy uses exactly the same group partition and the same synthetic batch; VAL and TEST are always
the public val.json and test.json (100 % real). Image paths are relative to DATA_ROOT (mounted as /data in
the containers): real images keep the `images/<source>/...` path of the annotation files, synthetic images
are `synthetic/<batch folder>/...` (see BATCHES).

Batch labels: v9 is the clean-background batch, v11 the copy-paste batch (`--lot`). Each call builds the
experiments of one batch; those of the other batches are not touched.

Experiments that reuse the data of another one (`data_from`) are produced by copy_seed_runs.py.

Usage:
  python prepare/prepare_strategies.py [--lot v9|v11] [--experiments exp06_synth100_real100 ...] [--rfdetr-root /data]
Outputs in PREPARE_OUT:
  <exp>/{train,val,test}.json                        RT-DETRv2 and DiffusionDet
  rfdetr/<exp>/{train,valid,test}/_annotations.coco.json   RF-DETR (absolute paths under --rfdetr-root)
  SUMMARY.json (clean-background batch) or SUMMARY_<lot>.json
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from prepare import strategies as st  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402

OUT = paths.PREPARE_OUT
# batch label -> (folder under DATA_ROOT/synthetic, sub-folder of the images relative to that folder)
# Each folder holds `synthetic_dataset.json` whose `file_name` is relative to <folder>/<sub-folder>.
BATCHES = {"v9": ("clean_background", "images"),
           "v11": ("copy_paste", ""),
           "v11_r25": ("copy_paste_r25", ""),
           "v11_r50": ("copy_paste_r50", ""),
           "p_pos_tight": ("proxy_p_pos_tight", "images"),
           "q_lab_same": ("proxy_q_lab_same", "images")}


def load_real():
    """Public train/val/test COCO files; `file_name` stays relative to DATA_ROOT (`images/...`)."""
    out = {}
    for s in paths.SPLITS:
        with open(paths.annotations_path(s), encoding="utf8") as f:
            ds = json.load(f)
        out[s] = {"info": {"description": f"DenSea-Aggregated ({s})"}, "licenses": [],
                  "categories": ds["categories"], "images": ds["images"], "annotations": ds["annotations"]}
    return out


def load_synth(lot, categories):
    """Synthetic batch aligned by class name to the real categories, without empty images.

    Returns ``(dataset, number of dropped images)``; `file_name` becomes relative to DATA_ROOT.
    """
    folder, sub = BATCHES[lot]
    path = paths.SYNTH_DIR / folder / "synthetic_dataset.json"
    if not path.exists():
        sys.exit(f"synthetic batch {lot} not found: {path}")
    with open(path, encoding="utf8") as f:
        ds = json.load(f)
    rel = "/".join(p for p in ("synthetic", folder, sub) if p)
    ref = {c["name"]: c["id"] for c in categories}
    remap = {c["id"]: ref[c["name"]] for c in ds["categories"] if c["name"] in ref}
    anns = [dict(a, category_id=remap[a["category_id"]]) for a in ds["annotations"] if a["category_id"] in remap]
    with_ann = {a["image_id"] for a in anns}
    images = [dict(im, file_name=f"{rel}/{im['file_name']}") for im in ds["images"] if im["id"] in with_ann]
    dropped = len(ds["images"]) - len(images)
    return {"info": ds.get("info", {}), "licenses": [], "categories": categories,
            "images": images, "annotations": anns}, dropped


def build(exp, real, synth, seed):
    """Training set of one experiment (see prepare/experiments.py)."""
    r = st.stratified_subset(real, exp["real"], seed) if exp["real"] > 0 else None
    mode, log = exp["synth"], {}
    if r is None:
        return st.merge(synth, {**synth, "images": [], "annotations": []}, exp["name"], exp.get("lot", "v9")), log
    if mode == "none":
        out = deepcopy(r)
        for im in out["images"]:
            im["source"] = "real"
        return out, log
    if mode == "all":
        part = synth
    elif mode == "minority_focus":
        part, log = st.minority_focus(r, synth, seed)
    else:
        part, log = st.inverse_proportional(r, synth, seed)
    return st.merge(part, r, exp["name"], exp.get("lot", "v9")), log


def rfdetr_copy(ds, root):
    out = deepcopy(ds)
    for im in out["images"]:
        im["file_name"] = f"{root.rstrip('/')}/{im['file_name']}"
    for c in out["categories"]:
        c["supercategory"] = "debris"
    return out


def write_exp(name, train, real, rf_root):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for s, ds in (("train", train), ("val", real["val"]), ("test", real["test"])):
        json.dump(ds, open(d / f"{s}.json", "w", encoding="utf8"))
        rd = OUT / "rfdetr" / name / ("valid" if s == "val" else s)
        rd.mkdir(parents=True, exist_ok=True)
        json.dump(rfdetr_copy(ds, rf_root), open(rd / "_annotations.coco.json", "w", encoding="utf8"))


def selected(args):
    """Experiments built by this script for the chosen batch.

    Phases 4, 7 and 8 have their own builders; experiments with `data_from` are copied by copy_seed_runs.py.
    """
    return {n: e for n, e in EXPERIMENTS.items()
            if (not args.experiments or n in args.experiments) and e["phase"] not in (4, 7, 8)
            and "data_from" not in e and e.get("lot", "v9") == args.lot}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--rfdetr-root", default="/data", help="DATA_ROOT inside the RF-DETR container")
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--lot", default="v9", choices=["v9", "v11"], help="synthetic batch label")
    args = ap.parse_args()
    real = load_real()
    exps = selected(args)
    synth, dropped = (None, 0)
    if any(e["synth"] != "none" for e in exps.values()):
        synth, dropped = load_synth(args.lot, real["train"]["categories"])
    summary = {}
    if synth:
        summary.update(synth_images=len(synth["images"]), synth_annotations=len(synth["annotations"]),
                       synth_dropped_empty=dropped)
    summary["experiments"] = {}
    for name, exp in exps.items():
        train, log = build({**exp, "name": name}, real["train"], synth, args.seed)
        write_exp(name, train, real, args.rfdetr_root)
        summary["experiments"][name] = {"images": len(train["images"]), "annotations": len(train["annotations"]),
                                        "synthetic_images": sum(im.get("source") == "synthetic" for im in train["images"]),
                                        "per_class": dict(st.class_counts(train)), "selection": log}
        print(name, summary["experiments"][name]["images"], "images")
    summary["lot"] = args.lot
    name = "SUMMARY.json" if args.lot == "v9" else f"SUMMARY_{args.lot}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / name, "w"), indent=1)


if __name__ == "__main__":
    main()
