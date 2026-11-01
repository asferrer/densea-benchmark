"""Common evaluation of all runs on TEST of the group partition, with paired bootstrap over images (B = 1000).

Input: a JSON manifest (template: eval/runs.json, written by eval/make_manifest.py) with
  {"runs": {"<name>": {"detector": "...", "strategy": "..."}},
   "pairs": [["<A>", "<B>"], ...]}          # differences A - B; default: each run - its Real-Only run
The predictions of a run are COCO results (image_id, category_id, bbox xywh, score) with the image ids of
the TEST annotations, read from `common.paths.predictions_dir(detector, "test") / <strategy>_test_preds.json`;
the 100 best detections per image are kept for all detectors. Runs without predictions are skipped.
Subsets: full, CleanSea, Roboflow (Neural_Ocean + Ocean_garbage).
Outputs (eval/out/): metrics.csv, bootstrap.csv, SUMMARY.md
TEST is evaluated once, with the frozen configuration: set ALLOW_TEST=1 to run it.
Usage: ALLOW_TEST=1 python eval/evaluate_runs.py [--manifest eval/runs.json] [--workers 8]
"""
import argparse
import csv
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from eval.coco_tools import Cache, ap_batch_cpu, load_dets, load_gt, official, preds_file  # noqa: E402

B = 1000
OUT = Path(__file__).resolve().parent / "out"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "runs.json"


def pred_file(run):
    return preds_file(run["detector"], run["strategy"], "test")


def subsets(gt):
    ids = sorted(gt.getImgIds())
    src = {i: gt.imgs[i].get("source", "CleanSea") for i in ids}
    return {"full": ids, "CleanSea": [i for i in ids if src[i] == "CleanSea"],
            "Roboflow": [i for i in ids if src[i] != "CleanSea"]}


def weights(name, ids, allids):
    """Same resamples for all runs (seed derived from the subset name)."""
    rng = np.random.default_rng(zlib.crc32(f"test/{name}".encode()))
    pos = np.searchsorted(allids, ids)
    w = np.zeros((B, len(allids)))
    w[:, pos] = rng.multinomial(len(pos), np.ones(len(pos)) / len(pos), size=B)
    w0 = np.zeros(len(allids)); w0[pos] = 1
    return w0, w


def work(job):
    """Official metrics and bootstrap AP/AP50/AP75 of one run on each subset."""
    name, run, gt_path = job
    gt = load_gt(gt_path)
    dt = load_dets(pred_file(run), gt)
    cache = Cache(gt, dt)
    allids = np.array(cache.img_ids)
    res = {}
    for sub, ids in subsets(gt).items():
        w0, w = weights(sub, ids, allids)
        r = {"n_img": len(ids), "official": official(gt, dt, ids, paths.CLASSES)}
        for met, t_idx in (("AP", None), ("AP50", [0]), ("AP75", [5])):
            r[met + "0"] = float(ap_batch_cpu(cache, w0[None], t_idx)[0][0])
            r[met + "b"] = ap_batch_cpu(cache, w, t_idx)[0]
            assert abs(r[met + "0"] - r["official"][met]) < 1e-6, (name, sub, met)
        res[sub] = r
    print("evaluated", name, flush=True)
    return name, res


def default_pairs(runs):
    ref = {r["detector"]: k for k, r in runs.items() if r["strategy"].startswith("exp07")}
    return [[k, ref[r["detector"]]] for k, r in runs.items()
            if r["detector"] in ref and k != ref[r["detector"]]]


def write_metrics(runs, res):
    rows = []
    for name, r in res.items():
        for sub, v in r.items():
            rows.append({"run": name, "detector": runs[name]["detector"], "strategy": runs[name]["strategy"],
                         "subset": sub, "n_img": v["n_img"],
                         **{k: round(100 * x, 2) for k, x in v["official"].items()},
                         **{f"{m}_ci": "[{:.2f}; {:.2f}]".format(*(100 * np.percentile(v[m + "b"], [2.5, 97.5])))
                            for m in ("AP", "AP50", "AP75")}})
    with open(OUT / "metrics.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def write_bootstrap(pairs, res):
    rows = []
    for a, b in pairs:
        for sub in res[a]:
            for met in ("AP", "AP50", "AP75"):
                d0 = res[a][sub][met + "0"] - res[b][sub][met + "0"]
                db = res[a][sub][met + "b"] - res[b][sub][met + "b"]
                lo, hi = np.percentile(db, [2.5, 97.5])
                p2 = min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))
                rows.append({"A": a, "B": b, "subset": sub, "metric": met, "n_img": res[a][sub]["n_img"],
                             "A_val": round(100 * res[a][sub][met + "0"], 2),
                             "B_val": round(100 * res[b][sub][met + "0"], 2), "diff": round(100 * d0, 2),
                             "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2), "p_two_sided": round(p2, 4)})
    if rows:
        with open(OUT / "bootstrap.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def write_summary(mrows, brows):
    md = ["# Evaluation on TEST of the group partition", "", "| Run | Subset | AP | AP50 | AP75 | 95% CI AP |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['run']} | {r['subset']} | {r['AP']} | {r['AP50']} | {r['AP75']} | {r['AP_ci']} |" for r in mrows]
    md += ["", "## Paired differences (bootstrap over images, B = 1000)", "",
           "| A − B | Subset | Metric | Difference | 95% CI | p |", "|---|---|---|---|---|---|"]
    md += [f"| {r['A']} − {r['B']} | {r['subset']} | {r['metric']} | {r['diff']:+.2f} | "
           f"[{r['ci_lo']:+.2f}; {r['ci_hi']:+.2f}] | {r['p_two_sided']} |" for r in brows]
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    man = json.load(open(args.manifest, encoding="utf8"))
    runs = {k: v for k, v in man["runs"].items() if pred_file(v).exists()}
    missing = sorted(set(man["runs"]) - set(runs))
    if missing:
        print("no predictions (skipped):", ", ".join(missing))
    gt_path = str(paths.annotations_path("test"))
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max(1, min(args.workers, len(runs)))) as ex:
        res = dict(ex.map(work, [(k, v, gt_path) for k, v in runs.items()]))
    pairs = [p for p in man.get("pairs") or default_pairs(runs) if p[0] in res and p[1] in res]
    write_summary(write_metrics(runs, res), write_bootstrap(pairs, res))
    print("written", OUT)


if __name__ == "__main__":
    main()
