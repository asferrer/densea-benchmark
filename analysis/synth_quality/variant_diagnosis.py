"""Diagnosis of the copy-paste gap (2/2): what still gives away the pasted objects of j_v11_visible (CPU only, TRAIN only).

Requires the features of `variant_features.py`. Compares the PASTED objects (P) with the real TRAIN objects recompressed
to JPEG q75 (Orc, main control: same compression pipeline as the generator), with the real objects not recompressed
(Oplain) and with the real objects kept in the same composed images (RC).
AUC = ResNet-50 + logistic regression (domain_gap recipe), CV grouped by image, mean of 3 seeds.
The "class-adjusted AUC" weights the real objects with the class mix of the evaluated subset of pasted objects.

Usage: python analysis/synth_quality/variant_diagnosis.py
Output: out/variant_diagnosis.md (+ .json with the figures).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variant_features as gf  # noqa: E402
import variant_diagnosis_strata as strata  # noqa: E402
from variant_diagnosis_stats import CLASSES, FAMILIES, STAT_KEYS, cv_auc, fmt, table, touches, wauc  # noqa: E402
from variant_diagnosis_text import LIMITATIONS, SUMMARY  # noqa: E402
from variant_quality import crop_quality  # noqa: E402

dg = gf.dg


# ------------------------------------------------------------------ loading
def load_all():
    pil, ref = gf.load_pilot(), gf.load_reference()
    L = lambda n, v="ctx": np.load(gf.DCACHE / f"{n}_{v}.npy")  # noqa: E731
    S = lambda n: json.load(open(gf.DCACHE / f"{n}_stats.json"))  # noqa: E731
    return {"pil": pil, "ref": ref, "F": {v: (L("P", v), L("Orc", v)) for v in gf.VARIANTS},
            "RC": L("RC"), "Oplain": L("Oplain"), "stats": {n: S(n) for n in ("P", "RC", "Orc", "Oplain")}}


def pair(D, fp, fo):
    """x, y, classes, groups for P (1) vs O (0)."""
    P, O = D["pil"]["P"], D["ref"]
    x = np.concatenate([fp, fo])
    y = np.r_[np.ones(len(fp)), np.zeros(len(fo))]
    cls = np.array([p["cls"] for p in P] + [o["cls"] for o in O])
    grp = np.array([p["group"] for p in P] + [o["group"] for o in O])
    return x, y, cls, grp


# ------------------------------------------------------------------ sections
def sec_global(D, base):
    P, RC = D["pil"]["P"], D["pil"]["RC"]
    x, y, cls, grp = base["x"], base["y"], base["cls"], base["grp"]
    sel = y == 1
    rows = [["P vs real TRAIN recompressed q75 (Orc) [main control]", fmt(base["cv"]["auc"], base["cv"]["std"]),
             fmt(*wauc(base["cv"]["oof"], y, cls, sel))]]
    xo, yo, co, go = pair(D, D["F"]["ctx"][0], D["Oplain"])
    r = cv_auc(xo, yo, go)
    rows.append(["P vs original real TRAIN (Oplain) [variant_pilot recipe]", fmt(r["auc"], r["std"]), fmt(*wauc(r["oof"], yo, co, yo == 1))])
    keep = [i for i, p in enumerate(P) if p["cls"] != "Metal_Debris"]
    xr = np.concatenate([D["F"]["ctx"][0][keep], D["RC"]])
    yr = np.r_[np.ones(len(keep)), np.zeros(len(RC))]
    cr = np.array([P[i]["cls"] for i in keep] + [o["cls"] for o in RC])
    gr = np.array([P[i]["group"] for i in keep] + [o["group"] for o in RC])
    r2 = cv_auc(xr, yr, gr)
    rows.append([f"P (without Metal_Debris) vs real kept in the same composites (RC, n={len(RC)})",
                 fmt(r2["auc"], r2["std"]), fmt(*wauc(r2["oof"], yr, cr, yr == 1))])
    out = {"orc": base["cv"]["auc"], "oplain": r["auc"], "rc": r2["auc"]}
    L = ["## 0. Global AUC per object (box with context)", ""]
    L += table(["Comparison", "AUC (grouped CV, 3 seeds)", "Class-adjusted AUC"], rows)
    L += ["", f"Control real TRAIN vs VAL (domain_gap, already measured): 0.636. "
          f"n pasted = {len(P)}, n real (Orc) = {len(D['ref'])}.", ""]
    return L, out


def sec_class(D, base):
    x, y, cls, grp = base["x"], base["y"], base["cls"], base["grp"]
    rows, out = [], {}
    for c in CLASSES:
        m = cls == c
        pc, oc = int(((cls == c) & (y == 1)).sum()), int(((cls == c) & (y == 0)).sum())
        a_g = dg.subgroup_auc(base["cv"]["oof"][m], y[m], ["x"] * int(m.sum())).get("x", {}).get("auc", float("nan"))
        r = cv_auc(x[m], y[m], grp[m]) if pc >= 10 and oc >= 10 else {"auc": float("nan"), "std": float("nan")}
        _, se = wauc(base["cv"]["oof"], y, cls, (y == 1) & (cls == c))
        rows.append([c, pc, oc, fmt(a_g, se), fmt(r["auc"], r["std"])])
        out[c] = {"auc_global": a_g, "auc_class": r["auc"], "n_pasted": pc}
    L = ["## 1. AUC per class", "", "\"Global model\": AUC inside the class of the out-of-fold probabilities of the 8-class "
         "classifier (± Hanley-McNeil standard error). \"Per-class model\": classifier trained only with that class.", ""]
    return L + table(["Class", "n pasted", "n real", "Global-model AUC", "Per-class-model AUC"], rows) + [""], out


def sec_region(D, base):
    P = D["pil"]["P"]
    rows, out = [], {}
    names = {"ctx": "(a) box with context", "inner": "(b) interior only (central 50 %)",
             "band": "(c) box border band only", "env": "(d) surroundings only (without the box)"}
    tp, to = touches(P), touches(D["ref"])
    keep = np.r_[~tp, ~to]
    for v in gf.VARIANTS:
        x, y, cls, grp = pair(D, *D["F"][v])
        r = cv_auc(x, y, grp)
        rb = cv_auc(x[keep], y[keep], grp[keep])
        out[v] = {"auc": r["auc"], "no_border": rb["auc"]}
        rows.append([names[v], fmt(r["auc"], r["std"]), fmt(*wauc(r["oof"], y, cls, y == 1)),
                     f"{r['auc'] - out['ctx']['auc']:+.3f}" if v != "ctx" else "-", fmt(rb["auc"], rb["std"])])
    L = ["## 2. AUC per region", "", "Same geometric regions in pasted and real objects (gray fill outside the region). "
         "The interior is the central rectangle (not the mask, which real objects do not have); in the surroundings the whole "
         "box is covered.", ""]
    L += table(["Region", "AUC", "Class-adjusted AUC", "Δ vs context", "AUC without boxes at the image border"], rows)
    L += ["", f"Pasted boxes at the image border (within 3 px): {tp.sum()} of {len(P)} pasted ({100 * tp.mean():.0f} %) vs "
          f"{to.sum()} of {len(to)} real TRAIN ({100 * to.mean():.0f} %) and {touches(D['pil']['RC']).mean() * 100:.0f} % of the real "
          f"objects kept in the same composites. The last column uses only objects that do not touch the border (n pasted = "
          f"{int((~tp).sum())}, n real = {int((~to).sum())}).", ""]
    return L, out


def sec_lowlevel(D):
    P, O, RC = D["stats"]["P"], D["stats"]["Orc"], D["stats"]["RC"]
    Op = D["stats"]["Oplain"]
    pc = np.array([p["cls"] for p in D["pil"]["P"]] + [o["cls"] for o in D["ref"]])
    y = np.r_[np.ones(len(P)), np.zeros(len(O))]
    col = lambda S, k: np.array([s[k] for s in S], float)  # noqa: E731
    rows, out = [], {}
    for k in STAT_KEYS:
        a, se = wauc(np.nan_to_num(np.r_[col(P, k), col(O, k)]), y, pc, y == 1)
        out[k] = a
        rows.append([k, f"{np.nanmedian(col(P, k)):.4g}", f"{np.nanmedian(col(RC, k)):.4g}", f"{np.nanmedian(col(O, k)):.4g}",
                     f"{np.nanmedian(col(Op, k)):.4g}", fmt(a, se), f"{abs(a - 0.5):.3f}"])
    fam = []
    grp = np.array([p["group"] for p in D["pil"]["P"]] + [o["group"] for o in D["ref"]])
    for name, ks in FAMILIES.items():
        x = np.nan_to_num(np.column_stack([np.r_[col(P, k), col(O, k)] for k in ks]))
        r = cv_auc(x, y, grp)
        fam.append([name, fmt(r["auc"], r["std"])])
        out["fam:" + name] = r["auc"]
    x = np.nan_to_num(np.column_stack([np.r_[col(P, k), col(O, k)] for k in STAT_KEYS]))
    rall = cv_auc(x, y, grp)
    out["all"] = rall["auc"]
    fam.append(["ALL the statistics together", fmt(rall["auc"], rall["std"])])
    L = ["## 3. Low-level statistics (interior region, central 50 % of the box)", "",
         "Medians: P pasted, RC real of the same composites, Orc real TRAIN recompressed q75, Oplain not recompressed. "
         "Univariate AUC (pasted = positive; 0.5 = does not separate; class-adjusted) against Orc. "
         "`noise_flat`, `lapvar_native` and `block` depend on the native resolution (CleanSea 1920x1080, Roboflow 416x416); "
         "`lapvar64` and `hf_ratio` are measured at 64x64.", ""]
    L += table(["Statistic", "P", "RC", "Orc", "Oplain", "Univ. AUC", "|AUC-0.5|"], rows)
    L += ["", "AUC of an LR classifier on each family (grouped CV):", ""] + table(["Family", "AUC"], fam) + [""]
    return L, out


# ------------------------------------------------------------------ report
def main():
    dg.lower_priority()
    D = load_all()
    x, y, cls, grp = pair(D, *D["F"]["ctx"])
    base = {"x": x, "y": y, "cls": cls, "grp": grp, "cv": cv_auc(x, y, grp)}
    P = D["pil"]["P"]
    cq = crop_quality(P, D["stats"]["P"])
    L = ["# Copy-paste gap diagnosis: what still gives away the pasted objects (j_v11_visible)", "",
         "TRAIN only. n pasted = %d (300 composed images over real TRAIN backgrounds); comparison real objects: %d TRAIN objects "
         "(domain_gap sample, recompressed to JPEG q75). CPU only; no VAL or TEST data except the TRAIN vs VAL control "
         "quoted in section 0." % (len(P), len(D["ref"])), ""]
    res = {}
    for key, fn in (("global", sec_global), ("class", sec_class), ("region", sec_region)):
        s, r = fn(D, base)
        L += s
        res[key] = r
    s, r = sec_lowlevel(D)
    L += s
    res["low_level"] = r
    s, r, cdom = strata.sec_domain(D, base)
    L += s
    res["domain"] = r
    s, r = strata.sec_quality(D, base, cq, cdom)
    L += s
    res["quality"] = r
    head, body = L[:4], L[4:]
    (dg.OUT / "variant_diagnosis.md").write_text("\n".join(head) + "\n" + SUMMARY + "\n" + "\n".join(body) + "\n" + LIMITATIONS,
                                                 encoding="utf8")
    json.dump(res, open(dg.OUT / "variant_diagnosis.json", "w", encoding="utf8"), indent=1, default=float)
    print("report written")


if __name__ == "__main__":
    main()
