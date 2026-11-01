"""Build the group partition (train/val/test of about 70/15/15) from the pooled annotations.

The three COCO files of ``--annotations-dir`` are pooled (their ``group`` field is ignored), images
that come from the same photograph or scene are linked into groups (``grouping.py``) and the groups
are assigned to partitions with a multi-label stratified procedure (``group_stratify.py``).

Content edges are recomputed from the descriptors in ``--features-dir`` (``dhash.npy`` and
``embeddings.npy``, rows in ascending image id) or, with ``--compute-features``, from the images under
``DATA_ROOT``. A pair is an edge if it meets the criterion in either direction.

Row order: the greedy assignment visits groups in order of their first row, so the result depends on the
row order. ``--order group`` (default) sorts the rows by the group id stored in the annotation files, which
reproduces the released group numbering and assignment exactly; ``--order id`` uses ascending image id and
needs no group field (it yields a different, equally valid partition of the same groups).

Usage (CPU, under 1 min once the descriptors exist):
  python split_group/make_split_group.py [--annotations-dir DIR] [--features-dir DIR] [--out DIR]
Outputs in ``--out``: split_ids.json, groups.json, report.json, REPORT.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from split_group.content_edges import content_pairs, features_exist, load_features  # noqa: E402
from split_group.group_stratify import group_matrix, split_groups  # noqa: E402
from split_group.grouping import build_groups  # noqa: E402
from split_group.pooling import pool_annotations  # noqa: E402
from split_group.report import SOURCES, write_report  # noqa: E402

RATIOS = (0.70, 0.15, 0.15)


def row_permutation(rows, order):
    """Indices that put the rows in the requested order (``id``: identity, ``group``: by group id)."""
    if order == "id":
        return np.arange(len(rows))
    if any(r["group"] is None for r in rows):
        sys.exit("--order group needs a `group` field in every image of the annotation files; use --order id")
    return np.array(sorted(range(len(rows)), key=lambda i: (rows[i]["group"], rows[i]["id"])))


def load_descriptors(args, rows):
    """Descriptors of ``rows`` (ascending image id), loaded from or computed into ``--features-dir``."""
    features_dir = Path(args.features_dir)
    if args.compute_features:
        from split_group import features
        hd, em = features.compute_for_annotations(args.annotations_dir, args.data_root, args.device)
        features.save_features(features_dir, hd, em)
    if not features_exist(features_dir):
        sys.exit(f"no descriptors in {features_dir}; run split_group/features.py or pass --compute-features")
    hd, em = load_features(features_dir)
    if len(hd) != len(rows) or len(em) != len(rows):
        sys.exit(f"descriptors have {len(hd)}/{len(em)} rows but the annotations have {len(rows)} images")
    return hd, em


def run_split(rows, cls, pairs, args):
    groups, gstats, _, _ = build_groups(rows, pairs, args.cleansea_mode)
    m = group_matrix(groups, cls, [r["source"] for r in rows], SOURCES, len(paths.CLASSES))
    totals = m[:, :len(paths.CLASSES)].sum(0)
    min_pc = np.floor(0.5 * min(RATIOS) * totals)            # at least half of the expected boxes in val/test
    assign, cur, cost, restart = split_groups(m, RATIOS, len(paths.CLASSES), seed=args.seed,
                                              restarts=args.restarts, min_per_class=min_pc)
    gstats.update({"cost": round(float(cost), 4), "best_restart": int(restart),
                   "min_per_class": min_pc.tolist(), "seed": args.seed, "restarts": args.restarts})
    return groups, assign, gstats


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-dir", default=str(paths.ANNOTATIONS_DIR))
    ap.add_argument("--features-dir", default=str(paths.FEATURES_DIR))
    ap.add_argument("--data-root", default=str(paths.DATA_ROOT), help="image root for --compute-features")
    ap.add_argument("--compute-features", action="store_true",
                    help="compute the descriptors from the images and save them in --features-dir")
    ap.add_argument("--order", default="group", choices=["id", "group"], help="row order (see module docstring)")
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--cleansea-mode", default="dive", choices=["dive", "dive_camera"])
    ap.add_argument("--td", type=int, default=24, help="dHash Hamming threshold (Roboflow sources only)")
    ap.add_argument("--tc", type=float, default=0.95, help="ResNet-50 cosine threshold")
    ap.add_argument("--tc-rf", type=float, default=0.86,
                    help="cosine threshold for Roboflow-Roboflow pairs (calibrated by visual inspection)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(paths.SPLIT_OUT))
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, cls = pool_annotations(args.annotations_dir)
    hd, em = load_descriptors(args, rows)
    perm = row_permutation(rows, args.order)
    rows, cls, hd, em = [rows[i] for i in perm], [cls[i] for i in perm], hd[perm], em[perm]
    is_cs = [r["source"] == "CleanSea" for r in rows]
    pairs, _, _ = content_pairs(hd, em, is_cs, td=args.td, tc=args.tc, device=args.device, tc_rf=args.tc_rf)
    print(f"content edges: {len(pairs)}")
    groups, assign, gstats = run_split(rows, cls, pairs, args)
    split_of = {rows[i]["id"]: paths.SPLITS[assign[g]] for g, members in enumerate(groups) for i in members}
    meta = {"seed": args.seed, "ratios": list(RATIOS), "cleansea_mode": args.cleansea_mode,
            "td": args.td, "tc": args.tc, "tc_rf": args.tc_rf}
    ids = {s: sorted(k for k, sp in split_of.items() if sp == s) for s in paths.SPLITS}
    json.dump({**ids, "meta": meta}, open(out / "split_ids.json", "w"))
    json.dump([{"group": g, "split": paths.SPLITS[assign[g]], "image_ids": sorted(rows[i]["id"] for i in mm)}
               for g, mm in enumerate(groups)], open(out / "groups.json", "w"))
    write_report(out, rows, cls, ids, gstats)
    print(json.dumps(gstats, indent=1))
    print({s: len(v) for s, v in ids.items()})


if __name__ == "__main__":
    main()
