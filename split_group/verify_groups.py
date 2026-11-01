"""Check that the group partition keeps related images together.

The partition is the one given by the annotation files (train.json, val.json, test.json). Between every
pair of partitions the script counts:
  1. normalised name keys present in both (flips, ``aN_`` prefixes, re-exports, same CleanSea file);
  2. CleanSea dives present in both (dive + camera, and the whole dive used by the partition);
  3. group ids present in both (needs the ``group`` field of the annotation files);
  4. content-similar pairs (dHash / ResNet-50 cosine criterion of the partition), recomputed from the
     descriptors in both directions; only when the descriptor files exist. Looser thresholds are
     reported as a sensitivity analysis and are not counted as violations.
Exit code 1 if checks 1-4 find any violation.

Usage:
  python split_group/verify_groups.py [--annotations-dir DIR] [--features-dir DIR] [--skip-content]
Outputs ``verification.json`` and ``verification.md`` in ``--out``.
"""
import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from split_group.content_edges import features_exist  # noqa: E402
from split_group.grouping import cleansea_sequence, name_key  # noqa: E402
from split_group.pooling import pool_annotations  # noqa: E402

PAIRS = [f"{paths.SPLITS[a]}-{paths.SPLITS[b]}" for a, b in itertools.combinations(range(3), 2)]
# (name, max dHash distance, min cosine with a CleanSea image, min cosine Roboflow-Roboflow)
SENSITIVITY = [("looser 1: dhash<=28, cos>=0.92 (CleanSea), cos>=0.84 (Roboflow)", 28, 0.92, 0.84),
               ("looser 2: dhash<=32, cos>=0.90 (CleanSea), cos>=0.82 (Roboflow)", 32, 0.90, 0.82)]


def shared_keys(keys, lab):
    """Keys present in more than one partition, and images carrying them, per pair of partitions."""
    by = collections.defaultdict(set)
    for k, s in zip(keys, lab):
        if k is not None:
            by[k].add(int(s))
    res = {}
    for a, b in itertools.combinations(range(3), 2):
        bad = {k for k, ss in by.items() if a in ss and b in ss}
        n_img = sum(1 for k, s in zip(keys, lab) if k in bad and s in (a, b))
        res[f"{paths.SPLITS[a]}-{paths.SPLITS[b]}"] = {"keys": len(bad), "images": n_img}
    return res


def cross_pairs(pairs, lab, is_cs, td, tc, tc_rf):
    """Cross-partition pairs meeting (dhash<=td, no CleanSea image) or cos>=tc or (Roboflow pair, cos>=tc_rf)."""
    res = collections.Counter()
    imgs = collections.defaultdict(set)
    for i, j, d, c in pairs:
        if lab[i] == lab[j]:
            continue
        rf = not (is_cs[i] or is_cs[j])
        if (d <= td and rf) or c >= tc or (rf and c >= tc_rf):
            a, b = sorted((int(lab[i]), int(lab[j])))
            key = f"{paths.SPLITS[a]}-{paths.SPLITS[b]}"
            res[key] += 1
            imgs[key].update((i, j))
    return {k: {"pairs": res[k], "images": len(imgs[k])} for k in res}


def content_checks(hd, em, lab, is_cs, td, tc, tc_rf, device="cpu"):
    """Criterion and sensitivity levels over the pairs found with the loosest thresholds."""
    from split_group.content_edges import content_pairs
    levels = [("partition criterion", td, tc, tc_rf)] + SENSITIVITY
    pairs, nn_d, nn_c = content_pairs(hd, em, is_cs, td=max(l[1] for l in levels), tc=min(l[2] for l in levels),
                                      device=device, tc_rf=min(l[3] for l in levels), cross_only=lab)
    res = {name: cross_pairs(pairs, lab, is_cs, t_d, t_c, t_rf) for name, t_d, t_c, t_rf in levels}
    nn = {}
    for s in range(3):
        sel = lab == s
        nn[paths.SPLITS[s]] = {"max_cosine_other_partition": round(float(nn_c[sel].max()), 4),
                               "min_dhash_other_partition_roboflow": int(nn_d[sel & ~is_cs].min())}
    return res, nn


def group_check(rows, lab):
    groups = [r["group"] for r in rows]
    if any(g is None for g in groups):
        return None
    return shared_keys(groups, lab)


def _cleansea_keys(rows, is_cs, mode):
    return [cleansea_sequence(r["file_name"], mode) if c else None for r, c in zip(rows, is_cs)]


def run_checks(rows, args, hd=None, em=None):
    """Return the result dictionary; ``violations`` counts checks 1-4."""
    lab = np.array([paths.SPLITS.index(r["file_split"]) for r in rows])
    is_cs = np.array([r["source"] == "CleanSea" for r in rows])
    res = {"images": {s: int((lab == i).sum()) for i, s in enumerate(paths.SPLITS)},
           "name_keys": shared_keys([name_key(r["file_name"], r["source"]) for r in rows], lab),
           "cleansea_dive_camera": shared_keys(_cleansea_keys(rows, is_cs, "dive_camera"), lab),
           "cleansea_dive": shared_keys(_cleansea_keys(rows, is_cs, "dive"), lab),
           "group_ids": group_check(rows, lab)}
    if hd is not None:
        res["content"], res["nearest_other_partition"] = content_checks(
            hd, em, lab, is_cs, args.td, args.tc, args.tc_rf, args.device)
    res["violations"] = count_violations(res)
    return res


def count_violations(res):
    total = 0
    for key in ("name_keys", "cleansea_dive_camera", "cleansea_dive", "group_ids"):
        total += sum(v["keys"] for v in (res.get(key) or {}).values())
    total += sum(v["pairs"] for v in res.get("content", {}).get("partition criterion", {}).values())
    return total


def _row(name, section, field):
    cells = [str((section or {}).get(p, {}).get(field, 0)) for p in PAIRS]
    return f"| {name} | " + " | ".join(cells) + " |"


def write_md(res, out_dir):
    head = ["| Check | " + " | ".join(PAIRS) + " |", "|---" * (len(PAIRS) + 1) + "|"]
    md = ["# Group partition verification", "",
          "Images: " + ", ".join(f"{k} {v}" for k, v in res["images"].items()) + f". Violations: **{res['violations']}**.", "",
          "Shared keys between partitions (0 means no key crosses partitions).", ""] + head
    md += [_row("normalised name keys", res["name_keys"], "keys"),
           _row("CleanSea dive + camera", res["cleansea_dive_camera"], "keys"),
           _row("CleanSea dive", res["cleansea_dive"], "keys")]
    if res["group_ids"] is not None:
        md.append(_row("group ids", res["group_ids"], "keys"))
    if "content" in res:
        md += ["", "Content-similar pairs across partitions.", "", *head]
        md += [_row(name, pairs, "pairs") for name, pairs in res["content"].items()]
        md += ["", "Nearest neighbour in another partition.", "",
               "| Partition | Max cosine | Min dHash (Roboflow) |", "|---|---|---|"]
        md += [f"| {s} | {v['max_cosine_other_partition']} | {v['min_dhash_other_partition_roboflow']} |"
               for s, v in res["nearest_other_partition"].items()]
    (Path(out_dir) / "verification.md").write_text("\n".join(md) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-dir", default=str(paths.ANNOTATIONS_DIR))
    ap.add_argument("--features-dir", default=str(paths.FEATURES_DIR))
    ap.add_argument("--skip-content", action="store_true", help="only checks 1-3")
    ap.add_argument("--td", type=int, default=24)
    ap.add_argument("--tc", type=float, default=0.95)
    ap.add_argument("--tc-rf", type=float, default=0.86)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(paths.SPLIT_OUT))
    args = ap.parse_args()
    rows, _ = pool_annotations(args.annotations_dir)
    hd = em = None
    features_dir = Path(args.features_dir)
    if not args.skip_content and features_exist(features_dir):
        from split_group.content_edges import load_features
        hd, em = load_features(features_dir)
        if len(hd) != len(rows):
            sys.exit(f"descriptors have {len(hd)} rows but the annotations have {len(rows)} images")
    elif not args.skip_content:
        print(f"no descriptors in {features_dir}: content check skipped")
    res = run_checks(rows, args, hd, em)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "verification.json", "w"), indent=1)
    write_md(res, out)
    print(json.dumps(res, indent=1))
    sys.exit(1 if res["violations"] else 0)


if __name__ == "__main__":
    main()
