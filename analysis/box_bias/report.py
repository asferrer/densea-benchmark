"""Tables and text of the box-bias report (box_bias_val.md) built from the results."""
import csv
from pathlib import Path

import numpy as np

from analysis.box_bias import box_bias_core as C
from common import paths

STATS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
CLASSES = paths.CLASSES
SHORT = {"exp07_real100": "exp07", "exp06_synth100_real100": "exp06",
         "exp08_real_synth_minor": "exp08", "exp09_real_synth_invprop": "exp09"}
EXTRA = ["iou", "abslog", "big"]
BREAK = ["ratio_area", "ratio_w", "ratio_h", "out_mean", "rel_mean"]


def f(x, d=2):
    return f"{x:.{d}f}"


def ci(s, d=3):
    return f"{f(s['median'], d)} [{f(s['median_ci'][0], d)}; {f(s['median_ci'][1], d)}]"


def cim(s, d=3):
    return f"{f(s['mean'], d)} [{f(s['mean_ci'][0], d)}; {f(s['mean_ci'][1], d)}]"


def bias_blocks(ms, sens, counts):
    """Summaries (mean and median with CI): overall and by class, COCO size and relative size."""
    for m in ms.values():
        m["out_mean"] = (m["out_l"] + m["out_t"] + m["out_r"] + m["out_b"]) / 4
        m["rel_mean"] = (m["rel_l"] + m["rel_t"] + m["rel_r"] + m["rel_b"]) / 4
        m["abslog"] = np.abs(np.log(m["ratio_area"]))
        m["big"] = (m["ratio_area"] > 1.3).astype(float)
    out = {"overall": {e: {k: C.summarize(m[k], m["gi"], counts) for k in C.METRICS + BREAK[3:] + EXTRA}
                       for e, m in ms.items()}}
    for name, key, vals in (("by_class", "cls", range(8)), ("by_size", "size", ["small", "medium", "large"]),
                            ("by_rsize", "rsize", ["<5 %", "5-15 %", ">15 %"])):
        out[name] = {str(v): {e: {k: C.summarize(m[k][m[key] == v], m["gi"][m[key] == v], counts)
                                  for k in BREAK} for e, m in ms.items()} for v in vals}
    out["sens"] = {str(t): {e: C.summarize(m["ratio_area"], m["gi"], counts) for e, m in d.items()}
                   for t, d in sens.items()}
    return out


def collect(corr):
    """Per (exp, variant): mean and SD over the splits of AP/AP50/AP75/APs before and after the correction."""
    by = {}
    for r in corr:
        by.setdefault((r["exp"], r["mode"], r["split"]), []).append(r)
    res = {}
    for (e, mode, s), rs in by.items():
        res.setdefault(e, {}).setdefault(mode, {})[s] = np.mean([x["stats"] for x in rs], 0)
    out = {}
    for e, d in res.items():
        out[e] = {}
        for mode in ("rel", "px"):
            delta = np.array([d[mode][s] - d["base"][s] for s in d["base"]])
            out[e][mode] = {"delta_mean": delta.mean(0).tolist(), "delta_sd": delta.std(0, ddof=1).tolist(),
                            "shift": [x["shift"] for x in corr if x["exp"] == e and x["mode"] == mode]}
        out[e]["base"] = np.mean([d["base"][s] for s in d["base"]], 0).tolist()
    return out


def table(rows, head):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def curves_md(data):
    cv = {e: {int(k): v for k, v in c.items()} for e, c in data["curves"].items()}
    ex = list(cv)
    best = {e: max(cv[e], key=lambda k: cv[e][k][0]) for e in ex}
    rows = []
    for e in ex:
        v = cv[e][best[e]]
        rows.append([SHORT[e], str(best[e])] + [f(x) for x in v] + [f(v[2] / v[1], 3)])
    out = table(rows, ["exp", "epoch"] + STATS + ["AP75/AP50"])
    d = [[SHORT[e]] + [f(cv[e][best[e]][i] - cv[ex[0]][best[ex[0]]][i], 2) for i in (0, 1, 2, 3, 4, 5, 8, 9)]
         for e in ex[1:]]
    out += ["", "Difference of the best checkpoint to exp07 (points):", ""]
    out += table(d, ["exp", "dAP", "dAP50", "dAP75", "dAPs", "dAPm", "dAPl", "dAR100", "dARs"])
    last = min(len(c) for c in cv.values())
    pl = []
    for e in ex:
        vs = np.array([cv[e][k] for k in range(41, last + 1)])
        m = vs.mean(0)
        pl.append([SHORT[e]] + [f(m[i]) for i in (0, 1, 2, 3, 8)] + [f(m[2] / m[1], 3)])
    out += ["", f"Mean per epoch between epochs 41 and {last} (range common to the four):", ""]
    out += table(pl, ["exp", "AP", "AP50", "AP75", "APs", "AR100", "AP75/AP50"])
    sel = [20, 40, 60, 70]
    out += ["", "AP on VAL per epoch:", ""]
    out += table([[SHORT[e]] + [f(cv[e][k][0]) if k in cv[e] else "-" for k in sel] for e in ex],
                 ["exp"] + [f"ep. {k}" for k in sel])
    return out, cv


def write_curves_csv(cv, path):
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["exp", "epoch"] + STATS + ["AP75_over_AP50"])
        for e, c in cv.items():
            for k, v in sorted(c.items()):
                w.writerow([e, k] + [round(x, 4) for x in v] + [round(v[2] / v[1], 4)])


def bias_md(data):
    b = data["bias"]
    ex = list(b["overall"])
    out = ["| exp | n pairs | area ratio, median [95% CI] | width ratio | height ratio | area, mean [CI] |",
           "|---|---|---|---|---|---|"]
    for e in ex:
        o = b["overall"][e]
        out.append(f"| {SHORT[e]} | {o['ratio_area']['n']} | {ci(o['ratio_area'])} | {ci(o['ratio_w'])} | "
                   f"{ci(o['ratio_h'])} | {cim(o['ratio_area'])} |")
    out += ["", "Shift per side (positive = the prediction protrudes from the GT), median [95% CI]:", ""]
    rows = [[SHORT[e]] + [ci(b["overall"][e][k], 1) for k in ("out_l", "out_t", "out_r", "out_b")] +
            [ci(b["overall"][e]["out_mean"], 1)] for e in ex]
    out += table(rows, ["exp", "left (px)", "top (px)", "right (px)", "bottom (px)", "mean of 4 sides (px)"])
    rows = [[SHORT[e]] + [ci(b["overall"][e][k], 4) for k in ("rel_l", "rel_t", "rel_r", "rel_b")] +
            [ci(b["overall"][e]["rel_mean"], 4)] for e in ex]
    out += ["", "Same, relative to the GT side (fraction):", ""]
    out += table(rows, ["exp", "left", "top", "right", "bottom", "mean of 4 sides"])
    out += ["", "Localisation quality of the pairs (median; mean for the fraction) [95% CI]:", ""]
    rows = [[SHORT[e], ci(b["overall"][e]["iou"]), cim(b["overall"][e]["iou"]), ci(b["overall"][e]["abslog"]),
             cim(b["overall"][e]["big"])] for e in ex]
    out += table(rows, ["exp", "IoU, median", "IoU, mean", "|ln area ratio|, median", "fraction with area ratio > 1.3"])
    out += ["", "Difference exp - exp07 (group bootstrap with the same resamples), 95% CI:", ""]
    rows = [[SHORT[e]] + [f"{f(b['overall'][e][m][s], 3)} - {f(b['overall'][ex[0]][m][s], 3)}; "
                          f"CI of diff. [{f(b['diff'][e][m][s][0], 3)}; {f(b['diff'][e][m][s][1], 3)}]"
                          for m, s in (("ratio_area", "median"), ("ratio_w", "median"), ("ratio_h", "median"))]
            for e in ex[1:]]
    out += table(rows, ["exp", "area (median)", "width (median)", "height (median)"])
    rows = [[SHORT[e], f"{f(b['overall'][e]['iou']['mean'], 4)} - {f(b['overall'][ex[0]]['iou']['mean'], 4)}; "
             f"CI of diff. [{f(b['diff'][e]['iou']['mean'][0], 4)}; {f(b['diff'][e]['iou']['mean'][1], 4)}]"] for e in ex[1:]]
    out += ["", "Mean IoU of the pairs, exp - exp07:", ""] + table(rows, ["exp", "mean IoU"])
    g = data["gt_counts"]
    out += ["", "GT of VAL by COCO size: " + ", ".join(f"{k} {v}" for k, v in g["size"].items()) +
            ". By relative size: " + ", ".join(f"{k} {v}" for k, v in g["rsize"].items()) + "."]
    return out + [""]


def break_md(data, name, title, labels):
    b = data["bias"][name]
    ex = list(data["bias"]["overall"])
    out = [f"{title}: median area ratio [95% CI] (n pairs) and mean shift per side (px)", ""]
    rows = []
    for k, lab in labels.items():
        row = [lab]
        for e in ex:
            s = b[k][e]["ratio_area"]
            row.append("-" if s is None else f"{ci(s)} (n={s['n']}); {f(b[k][e]['out_mean']['mean'], 1)}")
        rows.append(row)
    return out + table(rows, ["group"] + [SHORT[e] for e in ex]) + [""]


def corr_md(data):
    c = data["corr"]
    ex = list(c)
    out = []
    for mode, name in (("rel", "Relative (fraction of the box dimension)"), ("px", "Absolute (px)")):
        rows = []
        for e in ex:
            base, d, sd = c[e]["base"], c[e][mode]["delta_mean"], c[e][mode]["delta_sd"]
            rows.append([SHORT[e]] + [f"{f(base[i])} -> {f(base[i] + d[i])} ({d[i]:+.2f} ± {sd[i]:.2f})"
                                      for i in (0, 1, 2, 3)])
        out += [f"{name} correction. Mean over {len(c[ex[0]][mode]['shift']) // 2} partitions of the groups into two "
                "halves (fit on one, evaluate on the other, both directions); in parentheses, mean change ± SD "
                "across partitions:", ""]
        out += table(rows, ["exp", "AP", "AP50", "AP75", "APs"]) + [""]
    return out


def write(data, out_dir):
    out_dir = Path(out_dir)
    data["corr"] = collect(data["corr"])
    cmd, cv = curves_md(data)
    write_curves_csv(cv, out_dir / "curves.csv")
    cal = [[SHORT[e]] + [f(data["calib"][e]["mine"][i], 3) for i in (0, 1, 2)] +
           [f(data["calib"][e]["log"][i], 3) for i in (0, 1, 2)] for e in data["calib"]]
    sens = data["bias"]["sens"]
    md = ["# Box bias of RT-DETRv2 with the clean-background synthetic batch (v9; VAL only)", "",
          "## 1. Curves from log.txt (no inference)", ""] + cmd + [
          "", "## 2. Calibration: AP recomputed with pycocotools against log.txt", ""]
    md += table(cal, ["exp", "AP recomputed", "AP50 recomputed", "AP75 recomputed", "AP log", "AP50 log", "AP75 log"])
    md += ["", f"## 3. Size bias (predictions with score >= {data['score_thr']}, IoU >= 0.5, same class)", ""]
    md += bias_md(data)
    md += break_md(data, "by_size", "By COCO size of the GT", {"small": "small", "medium": "medium", "large": "large"})
    md += break_md(data, "by_rsize", "By relative size of the GT (square root of the box area / square root of the "
                   "image area)", {"<5 %": "<5 %", "5-15 %": "5-15 %", ">15 %": ">15 %"})
    md += break_md(data, "by_class", "By class", {str(i): c for i, c in enumerate(CLASSES)})
    md += ["Sensitivity to the score threshold (median area ratio [CI]):", ""]
    md += table([[SHORT[e]] + [ci(sens[t][e]) for t in sens] for e in data["calib"]],
                ["exp"] + [f"score >= {t}" for t in sens]) + [""]
    md += ["## 4. Post-hoc correction (shrink the boxes by the mean bias per side)", ""] + corr_md(data)
    md += READING
    (out_dir / "box_bias_val.md").write_text("\n".join(md) + "\n", encoding="utf8")


READING = """
## 5. Reading of the results (the tables above are authoritative)

- **Calibration.** The AP recomputed with pycocotools on the own predictions (CPU, fp32) differs from `log.txt`
  (GPU, AMP) by at most 0.05 points in AP, AP50 and AP75.
- **Curves.** On real VAL the three exps with synthetic v9 exceed exp07: AP +0.6 (exp06), +1.1 (exp08), +1.1 (exp09); AP50
  +1.7 to +2.0. AP75 rises +0.5, +1.9 and +1.6. The AP75/AP50 ratio is 0.731, 0.752 and 0.750 against 0.745 for exp07: only
  exp06 (epoch 55 of 74) is below, and there is no systematic loss in AP75.
- **Small objects.** APs neither rises nor falls appreciably (3.5, 2.6, 3.4 against 3.8), but VAL has 42 "small" GT
  (0.9 % of 4,554) and only 5 to 8 pairs per exp: the small-object hypothesis **cannot be tested on VAL**.
  ARs is much higher with synthetic data (26 to 32 against 11.5), with the same caveat.
- **Size bias.** Median pred/gt area ratio 0.994 to 0.998 in the four exps (exp07: 0.998); the differences to
  exp07 have CIs that include 0 and the median shift per side is 0.6 px at most (the margin of the synthetic boxes is
  ~20 px per side). There is no bias by class or by size: the only class with a ratio > 1 is Fishing_Net (1.04 to 1.05), and it
  is already present in exp07 (1.047), so it is not due to the synthetic data.
- **Localisation.** The only detectable effect: mean IoU of the pairs of exp06 0.830 against 0.837 for exp07 (CI of the
  difference [-0.013; -0.002]); exp08 and exp09 are indistinguishable from exp07.
- **Post-hoc correction.** Shrinking the boxes with the mean bias estimated on the other half does not recover AP (relative: -0.14,
  -0.12 and +0.03 in exp06, exp08 and exp09, against -0.20 in the control exp07; absolute in px, worse in all). The damage
  attributable to box size is indistinguishable from 0 (|change| <= 0.2 AP, of the order of the SD across partitions).
- **Caveats.** It is measured on real VAL images (the synthetic data does not enter); `best.pth` is selected by AP on that same
  VAL (it affects the four equally); matching requires IoU >= 0.5, so a strongly inflated box on a small object would
  have no pair (survivorship bias), but the sensitivity to score 0.1/0.3/0.5 is nil and the VAL objects are almost all
  large; it cannot be ruled out that the inflation harms small objects, which VAL does not contain.
""".split("\n")
