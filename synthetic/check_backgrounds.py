"""Check candidate background images against the whole corpus with the group-partition metric.

Background images may be frames or object-free versions of images of the corpus. Their descriptors are
computed with the recipe of ``split_group/features.py`` and compared with the descriptors of every corpus
image (``FEATURES_DIR``, rows in ascending image id of the annotation files). A background has a relative if
it meets the criterion of the partition: dHash <= 24 (against Roboflow images), cosine >= 0.95 (against
CleanSea images) or cosine >= 0.86 (against Roboflow images). Backgrounds with a relative in val or test are
not allowed; the cosine >= 0.90 neighbours in val/test are listed for review.

Usage:
  python synthetic/check_backgrounds.py --backgrounds-dir <folder> [--out DIR] [--device cuda]
Outputs in --out: backgrounds_check.json, allowed_backgrounds.txt (one file name per line), BACKGROUNDS.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from split_group.content_edges import features_exist, load_features  # noqa: E402
from split_group.features import compute  # noqa: E402
from split_group.pooling import pool_annotations  # noqa: E402

IMG = (".png", ".jpg", ".jpeg")


def match(bg_hd, bg_em, hd, em, is_cs):
    """Per background: minimum Hamming distance (both directions, CleanSea images excluded) and cosine to the corpus."""
    s_all = torch.from_numpy(hd.astype(np.float32) * 2 - 1)
    q = torch.from_numpy(bg_hd.astype(np.float32) * 2 - 1)
    d1 = ((256 - torch.einsum("qk,ntk->qnt", q[:, 0], s_all).amax(2)) / 2)
    d2 = ((256 - torch.einsum("nk,qtk->qnt", s_all[:, 0], q).amax(2)) / 2)
    d = torch.minimum(d1, d2).numpy()
    d[:, is_cs] = 999
    c = bg_em @ em.T
    return d, c


def summarize(names, d, c, rows, td=24, tc=0.95, tc_rf=0.86, tc_review=0.90):
    """One record per background with the partitions of its relatives (rows carry `file_split`)."""
    rf = np.array([r["source"] != "CleanSea" for r in rows])
    res = []
    for b, name in enumerate(names):
        hit = np.where((d[b] <= td) | (c[b] >= tc) | ((c[b] >= tc_rf) & rf))[0]
        rev = np.where(c[b] >= tc_review)[0]
        res.append({"background": name, "min_dhash": int(d[b].min()), "max_cos": round(float(c[b].max()), 4),
                    "nearest": rows[int(c[b].argmax())]["file_name"],
                    "splits_metric": sorted({rows[i]["file_split"] for i in hit}),
                    "splits_review": sorted({rows[i]["file_split"] for i in rev}), "n_metric": len(hit)})
    return res


def write_outputs(res, out, bg_dir):
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "backgrounds_check.json", "w"), indent=1)
    bad = [r for r in res if set(r["splits_metric"]) & {"val", "test"}]
    ok = [r["background"] for r in res if r not in bad]
    (out / "allowed_backgrounds.txt").write_text("\n".join(ok) + "\n", encoding="utf8")
    rev = [r for r in res if set(r["splits_review"]) & {"val", "test"} and r not in bad]
    md = ["# Backgrounds against the group partition", "",
          f"Folder: `{Path(bg_dir).name}`; {len(res)} backgrounds. With a relative in the corpus: "
          f"{sum(1 for r in res if r['n_metric'])}; with a relative in val or test: **{len(bad)}** (excluded "
          f"from `allowed_backgrounds.txt`, {len(ok)} allowed). Cosine >= 0.90 with val or test, not excluded: "
          f"{len(rev)}.", "", "| Background | min dHash | max cos | nearest image | partitions |",
          "|---|---|---|---|---|"]
    md += [f"| {r['background']} | {r['min_dhash']} | {r['max_cos']} | {Path(r['nearest']).name} | "
           f"{','.join(r['splits_metric'])} |" for r in bad + rev]
    (out / "BACKGROUNDS.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"backgrounds {len(res)}; with a relative in val/test {len(bad)}; for review {len(rev)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backgrounds-dir", required=True, help="folder with the candidate background images")
    ap.add_argument("--annotations-dir", default=str(paths.ANNOTATIONS_DIR))
    ap.add_argument("--features-dir", default=str(paths.FEATURES_DIR))
    ap.add_argument("--out", default="backgrounds_check")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not features_exist(Path(args.features_dir)):
        sys.exit(f"no corpus descriptors in {args.features_dir}; run split_group/features.py first")
    names = sorted(f for f in os.listdir(args.backgrounds_dir) if f.lower().endswith(IMG))
    bg_hd, bg_em = compute([f"{args.backgrounds_dir}/{n}" for n in names], args.device, log_every=100)
    rows, _ = pool_annotations(args.annotations_dir)
    hd, em = load_features(Path(args.features_dir))
    d, c = match(bg_hd, bg_em, hd, em, np.array([r["source"] == "CleanSea" for r in rows]))
    write_outputs(summarize(names, d, c, rows), Path(args.out), args.backgrounds_dir)


if __name__ == "__main__":
    main()
