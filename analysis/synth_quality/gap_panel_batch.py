"""Gap panel on the FINAL copy-paste batch (v11, 12,000 images) against its 300-image pilot.

The panel is otherwise measured on 300-image pilots (gap_panel_drop.md: p_pos_tight and s_drop). Here the pasted
objects of the full batch (only origin != 'real') are measured with the SAME per-class quotas as gap_panel_drop.json,
next to the pilot control and s_drop (gap_panel_drop caches) and the clean-background batch (ref_batch_v9, main cache)
as an anchor of "clearly worse". CV groups: partition group of the real background image (merged COCO field
`background`). Real TRAIN and VAL only.

The full batch is read from FULL_BATCH (default DATA_ROOT/synthetic/v11): `synthetic_dataset.json` and the images at
<FULL_BATCH>/<file_name>. Requires `gap_panel_drop.py` first (quotas and caches).

Usage (CPU, low priority): python analysis/synth_quality/gap_panel_batch.py [--extractors ...]
Output: out/gap_panel_batch.json and out/gap_panel_batch.md. Own cache: RUNS_DIR/synth_quality/cache/gap_panel_batch/.
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
import gap_extractors as gx  # noqa: E402
import gap_panel as gp  # noqa: E402
import gap_panel_drop as gd  # noqa: E402

av, dg, OUT = gp.av, gp.dg, gp.OUT
FULL_BATCH = Path(os.environ.get("FULL_BATCH", dg.paths.SYNTH_DIR / "v11"))
NAME = "full_batch_v11"
CACHE_BATCH = gx.CACHE_MAIN.parent / "gap_panel_batch"
ROWS = [f"drop_{gd.CONTROL}", f"drop_{gd.TEST}", NAME, gp.REF]
LABEL = {f"drop_{gd.CONTROL}": "pilot p_pos_tight", f"drop_{gd.TEST}": "pilot s_drop", NAME: "full batch",
         gp.REF: "clean backgrounds (v9)"}


def load_batch():
    """Images and PASTED annotations of the merged batch; paths = root / file_name (chunk_k/images/...)."""
    d = json.load(open(FULL_BATCH / "synthetic_dataset.json", encoding="utf8"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: {"id": im["id"], "path": str(FULL_BATCH / im["file_name"]), "source": "synthetic",
                       "w": im["width"], "h": im["height"], "background": im.get("background")} for im in d["images"]}
    anns = [{"img": a["image_id"], "bbox": a["bbox"], "cls": names[a["category_id"]]}
            for a in d["annotations"] if not a.get("iscrowd", 0) and a.get("origin", "pasted") != "real"]
    return imgs, anns


def batch_pack(items, imgs, ext):
    f = np.load(CACHE_BATCH / ext / f"{NAME}.npy")
    gm, _ = av.synth_image_groups(NAME, [imgs[i] for i in sorted({m["img"] for m in items})],
                                  av.background_group_map(), False)
    return {"f": f, "meta": items, "g": [gm[m["img"]] for m in items], "cls": [m["cls"] for m in items]}


def run(exts):
    quotas = json.load(open(OUT / "gap_panel_drop.json", encoding="utf8"))["quotas"]
    imgs, anns = load_batch()
    batch = dg.sample_objects(imgs, anns, quotas)[0]
    drop, _ = gd.build(gd.ROOT)
    ref_items = gp.real_sets()[gp.REF][0]
    out = {"quotas": quotas, "n_batch": len(batch), "per_extractor": {}}
    for ext in exts:
        with gx.cache_at(CACHE_BATCH):
            gx.extract_set(NAME, batch, [ext])
        packs = {NAME: batch_pack(batch, imgs, ext)}
        packs.update({n: gp.load_pack(n, it, ext) for n, (it, _) in gp.real_sets().items() if n.startswith("real_")})
        packs[gp.REF] = gp.load_pack(gp.REF, ref_items, ext)
        with gx.cache_at(gd.CACHE_DROP):
            packs.update({n: gp.load_pack(n, it, ext, flag=gd.CONTROL) for n, it in drop.items()})
        out["per_extractor"][ext] = gp.panel_scenario(ext, ROWS, quotas, packs)
    json.dump(out, open(OUT / "gap_panel_batch.json", "w", encoding="utf8"), indent=1, default=float)
    write_md(out)
    return out


def write_md(out):
    L = ["# Panel on the final copy-paste batch (v11) against its pilot", "",
         f"Quotas per class (those of gap_panel_drop): {out['quotas']}. Pasted objects of the batch: {out['n_batch']}.", "",
         "| Extractor | Set | Grouped AUC [CI] | KID | CMMD | Vendi | Coverage @val | Recall @val |",
         "|---|---|---|---|---|---|---|---|"]
    for ext, r in out["per_extractor"].items():
        for n in ROWS:
            g = r[n]["global"]
            L.append(f"| {ext} | {LABEL[n]} | {g['auc']:.3f} [{g['auc_ci'][0]:.3f}; {g['auc_ci'][1]:.3f}] | "
                     f"{g['kid']:.4f} | {g['cmmd']:.3f} | {g['vendi']:.1f} | {g['coverage@val']:.2f} | "
                     f"{g['recall@val']:.2f} |")
        s = r["_floor"]["global"]
        L.append(f"| {ext} | real floor (TRAIN vs VAL) | {s['auc']:.3f} [{s['auc_ci'][0]:.3f}; "
                 f"{s['auc_ci'][1]:.3f}] | {s['kid']:.4f} | {s['cmmd']:.3f} | {s['vendi']:.1f} | – | – |")
    L += ["", "Grouped AUC: 3 CV seeds by partition group; CI by group bootstrap (gap_panel_core). "
              "Same quotas in every row; the pilot and the batch come from the same generator (s_drop)."]
    Path(OUT / "gap_panel_batch.md").write_text("\n".join(L) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extractors", nargs="*", default=None)
    a = ap.parse_args()
    dg.lower_priority()
    os.environ.setdefault("GAP_THREADS", "6")
    exts, bad = gx.available(a.extractors or gx.ORDER)
    print("extractors:", exts, bad, flush=True)
    run(exts)


if __name__ == "__main__":
    main()
