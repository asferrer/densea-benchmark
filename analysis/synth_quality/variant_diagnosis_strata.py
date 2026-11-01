"""Sections 4 and 5 of the copy-paste gap diagnosis: crop domain, crop quality and scaling (strata and regression)."""
import numpy as np

import variant_features as gf
from variant_diagnosis_stats import cv_auc, fmt, logit, table, touches, wauc
from variant_quality import emulate_geometry


def strata_rows(P, base, crit):
    """Rows (criterion, n, % pasted, AUC inside, AUC outside, Δ, median logit inside/outside) for a list of masks."""
    y, cls, oof = base["y"], base["cls"], base["cv"]["oof"]
    n_p = len(P)
    rows = []
    for name, m in crit:
        full = np.r_[m, np.zeros(int((y == 0).sum()), bool)]
        a1, s1 = wauc(oof, y, cls, full)
        a0, s0 = wauc(oof, y, cls, np.r_[~m, np.zeros(int((y == 0).sum()), bool)])
        rows.append([name, int(m.sum()), f"{100 * m.mean():.0f} %", fmt(a1, s1), fmt(a0, s0),
                     f"{a1 - a0:+.3f}" if a1 == a1 and a0 == a0 else "n/a",
                     f"{np.median(logit(oof[:n_p][m])):+.2f} / {np.median(logit(oof[:n_p][~m])):+.2f}" if 0 < m.sum() < n_p else "-"])
    return rows


STRATA_HEAD = ["Criterion", "n", "% pasted", "AUC inside", "AUC outside", "Δ", "median logit inside/outside"]


def sec_domain(D, base):
    P = D["pil"]["P"]
    cd = gf.crop_domains()
    cdom = np.array([cd.get(gf.source_key(p["src"]), "?") for p in P])
    ddom = np.array([p["dest_dom"] for p in P])
    y, cls, oof = base["y"], base["cls"], base["cv"]["oof"]
    odom = np.array([""] * len(P) + [o["dest_dom"] for o in D["ref"]])
    doms = ["CleanSea", "Neural Ocean", "Ocean Garbage"]
    rows, out = [], {}
    for cdm in doms:
        for ddm in doms:
            m = (cdom == cdm) & (ddom == ddm)
            sel = np.r_[m, np.zeros(int((y == 0).sum()), bool)]
            keep = sel | ((y == 0) & (odom == ddm))
            a, se = wauc(oof[keep], y[keep], cls[keep], sel[keep]) if m.sum() >= 5 else (float("nan"), float("nan"))
            rows.append([cdm, ddm, "same" if cdm == ddm else "different", int(m.sum()), fmt(a, se),
                         f"{np.median(logit(oof[:len(P)][m])):+.2f}" if m.any() else "-"])
            out[f"{cdm}->{ddm}"] = {"n": int(m.sum()), "auc": a}
    same, cross = cdom == ddom, cdom != ddom
    rob = lambda a: np.isin(a, ["Neural Ocean", "Ocean Garbage"])  # noqa: E731
    crit = [("crop domain = target domain", same), ("crop domain != target domain", cross),
            ("Roboflow crop -> CleanSea target", rob(cdom) & (ddom == "CleanSea")),
            ("CleanSea crop -> Roboflow target", (cdom == "CleanSea") & rob(ddom)),
            ("Roboflow crop -> Roboflow target (different subdomain)", rob(cdom) & rob(ddom) & (cdom != ddom))]
    L = ["## 4. Domain of the crop's origin vs domain of the target image", "",
         "AUC of the pasted objects of each cell against the real objects of the SAME target domain (class-adjusted, "
         "out-of-fold probabilities of model 0). Cells with n < 15 are noisy.", ""]
    L += table(["Crop domain", "Target domain", "Relation", "n", "AUC", "median logit"], rows)
    L += ["", "Aggregates (AUC inside / outside the criterion against all the real objects):", ""]
    L += table(STRATA_HEAD, strata_rows(P, base, crit)) + [""]
    return L, out, cdom


def quality_criteria(D, cq):
    """Masks over the pasted objects for the crop-quality and scaling criteria (name, bool mask)."""
    P = D["pil"]["P"]
    sc = np.array([p["scale"] for p in P], float)
    cmin = np.array([q["crop_min"] for q in cq], float)
    halo = np.array([q["halo"] for q in cq], float)
    mism = np.array([q["bg_mismatch"] for q in cq], float)
    rel = np.array([p["target_rel"] or np.nan for p in P], float)
    ddom = np.array([p["dest_dom"] for p in P])
    med_h, med_m = np.nanmedian(halo), np.nanmedian(mism)
    SP, SO = D["stats"]["P"], D["stats"]["Orc"]
    hf, de = np.array([t["hf_ratio"] for t in SP]), np.array([t["contrast_dE"] for t in SP])
    hf_p25, de_p25 = np.percentile([t["hf_ratio"] for t in SO], 25), np.percentile([t["contrast_dE"] for t in SO], 25)
    return [(f"blurrier than the real p25 (hf_ratio < {hf_p25:.4f})", hf < hf_p25),
            (f"low local contrast, < real p25 (contrast_dE < {de_p25:.0f})", de < de_p25),
            ("box at the image border (<= 3 px)", touches(P)), ("scale > 1.0 (enlarged crop)", sc > 1.0), ("scale > 1.5", sc > 1.5),
            ("scale < 0.5 (strongly reduced)", sc < 0.5),
            ("original crop < 40 px on its smaller side", cmin < 40), ("effective resolution (smaller side x scale) < 40 px", cmin * sc < 40),
            ("rectangular crop (alpha covers > 95 %): original background included", np.array([q["rect"] for q in cq], bool)),
            (f"border halo > median ({med_h:.2f})", np.nan_to_num(halo, nan=-9) > med_h),
            ("border halo > 0.5 (border ~ original background color)", np.nan_to_num(halo, nan=-9) > 0.5),
            (f"original-background vs target color mismatch > median ({med_m:.0f})", np.nan_to_num(mism, nan=-9) > med_m),
            ("relative target size > p75", rel > np.nanpercentile(rel, 75)), ("relative target size < p25", rel < np.nanpercentile(rel, 25)),
            ("CleanSea target (1920x1080, video)", ddom == "CleanSea")]


def sec_quality(D, base, cq, cdom):
    P = D["pil"]["P"]
    sc = np.array([p["scale"] for p in P], float)
    cmin = np.array([q["crop_min"] for q in cq], float)
    halo = np.array([q["halo"] for q in cq], float)
    mism = np.array([q["bg_mismatch"] for q in cq], float)
    rect = np.array([q["rect"] for q in cq], bool)
    rel = np.array([p["target_rel"] or np.nan for p in P], float)
    ddom = np.array([p["dest_dom"] for p in P])
    tb = touches(P)
    crit = quality_criteria(D, cq)
    L = ["## 5. Crop quality and scaling", "",
         "AUC of the pasted objects that meet the criterion against the real objects with the same class mix, and of those "
         "that do not (model 0). Δ > 0: the criterion makes them MORE detectable.", ""]
    L += table(["Criterion", "n pasted affected"] + STRATA_HEAD[2:], strata_rows(P, base, crit)) + [""]
    bins = [("< 0.5", sc < 0.5), ("0.5 - 0.8", (sc >= 0.5) & (sc < 0.8)), ("0.8 - 1.2", (sc >= 0.8) & (sc < 1.2)),
            ("1.2 - 2", (sc >= 1.2) & (sc < 2)), (">= 2", sc >= 2)]
    L += ["By scale-factor range:", ""] + table(["Scale", "n", "% pasted", "AUC inside", "AUC outside", "Δ", "median logit"],
                                                 strata_rows(P, base, bins)) + [""]
    halo_f = np.where(np.isfinite(halo), halo, np.nanmedian(halo))   # rectangular crops: no measurable halo -> median
    mism_f = np.where(np.isfinite(mism), mism, np.nanmedian(mism))
    L += scale_stats(D, sc) + sharp_source(D, cq)
    L += residual(D, base, tb, sc)
    cov = {"log scale": np.log(sc), "log smaller crop side": np.log(cmin), "rectangular crop": rect.astype(float),
           "halo": halo_f, "background color mismatch": mism_f, "relative target size": rel, "touches the border": tb.astype(float),
           "cross domain": (cdom != ddom).astype(float)}
    for k in ("hf_ratio", "noise_flat", "contrast_dE", "block", "sat"):
        cov[k] = np.array([s[k] for s in D["stats"]["P"]], float)
    L += regression(P, base, cov)
    scale_gt1 = np.r_[sc > 1, np.zeros(int((base['y'] == 0).sum()), bool)]
    return L, {"scale>1": float(wauc(base["cv"]["oof"], base["y"], base["cls"], scale_gt1)[0])}


def scale_stats(D, sc):
    """Medians of sharpness and contrast of the pasted objects per scale range against the real ones."""
    S = D["stats"]["P"]
    O = D["stats"]["Orc"]
    keys = ("hf_ratio", "lapvar64", "noise_flat", "contrast_dE", "l_std")
    rows = [["real Orc"] + [f"{np.nanmedian([s[k] for s in O]):.4g}" for k in keys] + [len(O)]]
    for name, m in (("pasted scale < 0.5", sc < 0.5), ("pasted 0.5 - 1.2", (sc >= 0.5) & (sc < 1.2)), ("pasted >= 1.2", sc >= 1.2)):
        rows.append([name] + [f"{np.nanmedian([s[k] for s, mm in zip(S, m) if mm]):.4g}" for k in keys] + [int(m.sum())])
    return ["Sharpness and contrast per scale range (medians):", ""] + table(["Group", *keys, "n"], rows) + [""]


def sharp_source(D, cq):
    """Is the sharpness loss due to the source crop or to the generator? Same statistic on the PNG and on the composite."""
    S = D["stats"]["P"]
    src_hf, src_lv = np.array([q["src_hf"] for q in cq]), np.array([q["src_lapvar64"] for q in cq])
    hf, lv = np.array([t["hf_ratio"] for t in S]), np.array([t["lapvar64"] for t in S])
    O = D["stats"]["Orc"]
    rows = [["source crop (PNG, central 50 %, before pasting)", f"{np.median(src_hf):.4f}", f"{np.median(src_lv):.1f}"],
            ["pasted in the composite (same region, after scale, rotation, blur, harmonization and JPEG)", f"{np.median(hf):.4f}", f"{np.median(lv):.1f}"],
            ["real TRAIN (Orc)", f"{np.median([t['hf_ratio'] for t in O]):.4f}", f"{np.median([t['lapvar64'] for t in O]):.1f}"],
            ["median composite / source ratio (per object)", f"{np.median(hf / src_hf):.2f}", f"{np.median(lv / src_lv):.2f}"]]
    em = emulate_geometry(D["pil"]["P"])
    st = ["source", "+ Gaussian pre-blur (crops < 100 px)", "+ rotation ±45° (cubic)", "+ scale (AREA / LANCZOS)"]
    erows = [[f"emulation: {n}", f"{np.nanmedian(em[:, i, 0]):.4f}", f"{np.nanmedian(em[:, i, 1]):.1f}"] for i, n in enumerate(st)]
    rows = rows[:1] + erows[1:] + rows[1:]
    return ["Is the sharpness loss due to the crop or to the generator? (statistics at 64x64)", "",
            "The \"emulation\" rows replicate the deterministic stages of `_transform_object` on the crop of each pasted object "
            "(random rotation, recorded scale); the rest of the drop down to the row \"pasted in the composite\" comes from "
            "harmonization, blur matching to the background, lighting, caustics/veil, alpha composition and JPEG q75.", ""] + table(
        ["What is measured", "median hf_ratio", "median lapvar64"], rows) + [""]


def residual(D, base, tb, sc):
    """AUC that remains after removing the giveaway cases (image border, extreme scale) from the pasted objects and the border ones from the real ones."""
    O = D["ref"]
    rows = []
    SP, SO = D["stats"]["P"], D["stats"]["Orc"]
    sharp = np.array([t["hf_ratio"] >= np.percentile([u["hf_ratio"] for u in SO], 25) and t["contrast_dE"] >= np.percentile(
        [u["contrast_dE"] for u in SO], 25) for t in SP])
    for name, mp, use_o in (("without boxes pasted at the border", ~tb, True),
                            ("without border and scale 0.5 - 1.5", ~tb & (sc >= 0.5) & (sc <= 1.5), True),
                            ("without border, scale 0.5 - 1.5, sharpness and contrast >= real p25", ~tb & (sc >= 0.5) & (sc <= 1.5) & sharp, True),
                            ("only sharpness and contrast >= real p25 (border and scale untouched)", sharp, False)):
        keep = np.r_[mp, ~touches(O) if use_o else np.ones(len(O), bool)]
        r = cv_auc(base["x"][keep], base["y"][keep], base["grp"][keep])
        rows.append([name, int(mp.sum()), int((keep.sum() - mp.sum())), fmt(r["auc"], r["std"]),
                     fmt(*wauc(r["oof"], base["y"][keep], base["cls"][keep], base["y"][keep] == 1))])
    return ["Residual AUC (box with context) after removing the giveaway cases:", ""] + table(
        ["Subset", "n pasted", "n real", "AUC", "Class-adjusted AUC"], rows) + [""]


def regression(P, base, cov):
    """Standardized OLS of the out-of-fold logit of each pasted object on the covariates (measures detectability)."""
    from scipy.stats import spearmanr
    z = logit(base["cv"]["oof"][:len(P)])
    cov = {k: v for k, v in cov.items() if np.nanstd(v) > 0}
    names = list(cov)
    X = np.column_stack([cov[k] for k in names])
    ok = np.isfinite(X).all(1)
    Xs = (X[ok] - X[ok].mean(0)) / (X[ok].std(0) + 1e-9)
    zs = (z[ok] - z[ok].mean()) / z[ok].std()
    A = np.column_stack([np.ones(ok.sum()), Xs])
    beta, *_ = np.linalg.lstsq(A, zs, rcond=None)
    res = zs - A @ beta
    cov_b = np.linalg.pinv(A.T @ A) * (res @ res / (len(zs) - A.shape[1]))
    rows = []
    for i, k in enumerate(names):
        rho = spearmanr(X[ok][:, i], z[ok])[0]
        rows.append([k, f"{rho:+.2f}", f"{beta[i + 1]:+.2f}", f"{beta[i + 1] / np.sqrt(cov_b[i + 1, i + 1]):+.1f}"])
    r2 = 1 - (res @ res) / (zs @ zs)
    return ["What explains the detectability of each pasted object (logit of the out-of-fold score of model 0; n = "
            f"{int(ok.sum())}; R² = {r2:.2f}):", ""] + table(["Covariate", "Spearman", "standardized coef.", "t"], rows) + [""]
