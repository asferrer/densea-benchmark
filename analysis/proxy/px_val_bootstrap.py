"""Transfer check of the generator: AP on VAL of the px_* runs (RT-DETRv2 trained on 10% of the real data, alone or
plus a synthetic batch) and paired differences with 95% CIs from a GROUP bootstrap over the group partition (the
images of a group enter or leave a resample together). VAL only, never TEST.

px_v9_*: clean-background batch; px_p_*, px_q_*: copy-paste variants; _s2: training seed 22.
Also hosts the helpers shared by the other bootstrap scripts: `group_weights`, `evaluate`, `METRICS`.
Predictions: CPU inference of best.pth (analysis/box_bias/infer_val_cpu.py), read from
`predictions_dir("rtdetrv2", "val") / <exp>_val_preds.json`.
Usage: python analysis/proxy/px_val_bootstrap.py [--preds DIR] [--b 2000]
Outputs: analysis/proxy/out/px_val.md and px_val.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
from common import paths  # noqa: E402
from eval.coco_tools import Cache, ap_batch_cpu, load_dets, load_gt, official  # noqa: E402

EXPS = ["px_real10", "px_v9_real10", "px_p_real10", "px_q_real10", "px_v9_real10_s2", "px_p_real10_s2"]
PAIRS = [("px_v9_real10", "px_real10"), ("px_p_real10", "px_real10"), ("px_q_real10", "px_real10"),
         ("px_p_real10", "px_v9_real10"), ("px_q_real10", "px_v9_real10"), ("px_q_real10", "px_p_real10"),
         ("px_v9_real10_s2", "px_real10"), ("px_p_real10_s2", "px_real10"), ("px_p_real10_s2", "px_v9_real10_s2")]
# Mean of seeds (21 and 22): the same group resample for the two runs of each arm.
SEED_MEANS = {"v9 (mean s21+s22)": ["px_v9_real10", "px_v9_real10_s2"],
              "p_pos_tight (mean s21+s22)": ["px_p_real10", "px_p_real10_s2"], "real (s21)": ["px_real10"]}
SEED_PAIRS = [("p_pos_tight (mean s21+s22)", "v9 (mean s21+s22)"), ("p_pos_tight (mean s21+s22)", "real (s21)"),
              ("v9 (mean s21+s22)", "real (s21)")]
METRICS = (("AP", None), ("AP50", [0]), ("AP75", [5]))
DIFF_HEAD = ["| A − B | Metric | Difference | 95% CI | p (two-sided) |", "|---|---|---|---|---|"]
OUT = Path(__file__).resolve().parent / "out"


def group_weights(img_ids, b, seed=21, groups_file=None):
    """Weights (b, n_img): each image weighs as many times as its group is drawn in the group resample.

    Groups come from `split_group/out/groups.json` (list of {"group", "split", "image_ids"}) unless
    `groups_file` is given. Returns the weights and the number of groups."""
    groups_file = groups_file or paths.SPLIT_OUT / "groups.json"
    g_of = {i: g["group"] for g in json.load(open(groups_file, encoding="utf8")) for i in g["image_ids"]}
    groups = np.array([g_of[i] for i in img_ids])
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(len(uniq), np.ones(len(uniq)) / len(uniq), size=b)
    return counts[:, inv].astype(float), len(uniq)


def evaluate(exp, gt, preds, w, split="val"):
    """Official metrics, point AP/AP50/AP75 (`<m>0`) and bootstrap AP/AP50/AP75 (`<m>b`) of `exp`.

    `preds` is the folder with `<exp>_<split>_preds.json`."""
    dt = load_dets(Path(preds) / f"{exp}_{split}_preds.json", gt)
    cache = Cache(gt, dt)
    assert list(cache.img_ids) == sorted(gt.getImgIds())
    off = official(gt, dt, cache.img_ids, paths.CLASSES)
    res = {"official": off}
    for m, t in METRICS:
        a0 = float(ap_batch_cpu(cache, np.ones((1, len(cache.img_ids))), t)[0][0])
        assert abs(a0 - off[m]) < 1e-6, (exp, m, a0, off[m])
        res[m + "0"], res[m + "b"] = a0, ap_batch_cpu(cache, w, t)[0]
    return res


def seed_mean_rows(res, out):
    """Differences between seed means per arm, with the same group resamples."""
    arms = {k: v for k, v in SEED_MEANS.items() if all(e in res for e in v)}
    rows = ["", "## Seed means per arm", ""] + DIFF_HEAD
    for x, y in [q for q in SEED_PAIRS if q[0] in arms and q[1] in arms]:
        for m, _ in METRICS:
            mean0 = {k: np.mean([res[e][m + "0"] for e in arms[k]]) for k in (x, y)}
            meanb = {k: np.mean([res[e][m + "b"] for e in arms[k]], axis=0) for k in (x, y)}
            d0, db = 100 * (mean0[x] - mean0[y]), 100 * (meanb[x] - meanb[y])
            lo, hi = np.percentile(db, [2.5, 97.5])
            p = min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))
            rows.append(f"| {x} − {y} | {m} | {d0:+.2f} | [{lo:+.2f}; {hi:+.2f}] | {p:.4f} |")
            out.setdefault("seed_means", []).append({"A": x, "B": y, "metric": m, "diff": round(d0, 2),
                                                     "ci": [round(lo, 2), round(hi, 2)], "p": round(p, 4)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=str(paths.predictions_dir("rtdetrv2", "val")),
                    help="folder with <exp>_val_preds.json")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    gt = load_gt(paths.annotations_path("val"))
    ids = sorted(gt.getImgIds())
    w, n_groups = group_weights(ids, a.b)
    exps = [e for e in EXPS if (Path(a.preds) / f"{e}_val_preds.json").exists()]
    res = {e: evaluate(e, gt, a.preds, w) for e in exps}
    md = [f"# Transfer check: AP on VAL (n = {len(ids)} images, {n_groups} groups of the group partition; "
          f"group bootstrap, B = {a.b})", "",
          "| Run | AP | AP50 | AP75 | APs | APm | APl | 95% CI AP |", "|---|---|---|---|---|---|---|---|"]
    for e in exps:
        o, lo_hi = res[e]["official"], 100 * np.percentile(res[e]["APb"], [2.5, 97.5])
        md.append(f"| {e} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75", "APs", "APm", "APl"))
                  + f" | [{lo_hi[0]:.2f}; {lo_hi[1]:.2f}] |")
    md += ["", "## Paired differences (same group resamples)", ""] + DIFF_HEAD
    out = {"n_img": len(ids), "n_groups": n_groups, "B": a.b, "runs": {}, "pairs": []}
    for e in exps:
        out["runs"][e] = {k: round(100 * v, 2) for k, v in res[e]["official"].items() if not isinstance(v, dict)}
    for x, y in [q for q in PAIRS if q[0] in res and q[1] in res]:
        for m, _ in METRICS:
            d0 = 100 * (res[x][m + "0"] - res[y][m + "0"])
            db = 100 * (res[x][m + "b"] - res[y][m + "b"])
            lo, hi = np.percentile(db, [2.5, 97.5])
            p = min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))
            md.append(f"| {x} − {y} | {m} | {d0:+.2f} | [{lo:+.2f}; {hi:+.2f}] | {p:.4f} |")
            out["pairs"].append({"A": x, "B": y, "metric": m, "diff": round(d0, 2), "ci": [round(lo, 2), round(hi, 2)],
                                 "p": round(p, 4)})
    md += seed_mean_rows(res, out)
    md += ["", "The CI of each run reflects the variation of VAL, not of the seed; the seed mean reduces the latter",
           "but with 2 seeds it does not estimate it (the difference between seeds of the same arm is the noise reference)."]
    OUT.mkdir(exist_ok=True)
    (OUT / "px_val.md").write_text("\n".join(md) + "\n", encoding="utf8")
    json.dump(out, open(OUT / "px_val.json", "w", encoding="utf8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
