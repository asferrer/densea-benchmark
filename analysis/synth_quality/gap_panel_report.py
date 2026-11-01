"""Report `out/gap_panel.md` from `out/gap_panel.json` and `out/gap_panel_validation.json`.

The summary of the findings lives in `out/gap_panel_summary.md` (when it exists it is placed first) so that regenerating
the tables does not erase it.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import gap_panel_core as gc  # noqa: E402
from common import paths  # noqa: E402

OUT = HERE / "out"
CLASSES = paths.CLASSES
NAMES = {"_floor": "real TRAIN vs VAL (floor)", "external_seabed": "Synthetic Seabed (7 classes)"}


def f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fci(g, k, d=3):
    ci = g.get(k + "_ci")
    return f"{f(g.get(k), d)} [{f(ci[0], d)}; {f(ci[1], d)}]" if ci else f(g.get(k), d)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def names(sc):
    return [k for k in sc if not k.startswith("_")]


def main_tables(ext, sc):
    ks = ["_floor"] + names(sc)
    g = lambda k: sc[k]["global"]  # noqa: E731
    t1 = table(["Variant", "n", "Grouped AUC ± sd [group CI]", "FD", "KID", "CMMD", "Precision", "Recall", "Density",
                "Coverage", "Vendi", "NN dist", "TSTR bacc"],
               [[NAMES.get(k, k), g(k).get("n_fake"), f"{f(g(k)['auc'])} ± {f(g(k)['auc_std'])} "
                 f"[{f(g(k)['auc_ci'][0])}; {f(g(k)['auc_ci'][1])}]", f(g(k)["fd"], 1), fci(g(k), "kid", 4),
                 f(g(k)["cmmd"], 3), fci(g(k), "precision", 2), fci(g(k), "recall", 2), fci(g(k), "density", 2),
                 fci(g(k), "coverage", 2), f(g(k)["vendi"], 1), f(g(k)["nn_dist"], 3),
                 f"{f(g(k)['tstr_bacc'], 3)} [{f(g(k)['tstr_bacc_ci'][0], 2)}; {f(g(k)['tstr_bacc_ci'][1], 2)}]"]
                for k in ks])
    t2 = table(["Variant", "FD", "KID", "CMMD", "Precision", "Recall", "Density", "Coverage", "NN synth→real"],
               [[NAMES.get(k, k), f(g(k)["fd@val"], 1), f(g(k)["kid@val"], 4), f(g(k)["cmmd@val"], 3),
                 f(g(k)["precision@val"], 2), f(g(k)["recall@val"], 2), f(g(k)["density@val"], 2),
                 f(g(k)["coverage@val"], 2), f(g(k)["nn_real"], 3)] for k in ks])
    return t1, t2


def class_table(sc, metric, d=2):
    ks = ["_floor"] + names(sc)
    return table(["Variant"] + CLASSES, [[NAMES.get(k, k)] + [f(sc[k]["per_class"].get(c, {}).get(metric), d)
                                                               for c in CLASSES] for k in ks])


def rank_section(res, metric, exts):
    """Rank of each variant per extractor (1 = best quality) and mean Spearman between extractors."""
    variants = [k for k in names(res["main"][exts[0]]) if k != "external_seabed"]
    ranks = {}
    for e in exts:
        vals = {k: res["main"][e][k]["global"][metric] for k in variants}
        order = gc.rank_order(vals, gc.sense_of(metric))
        ranks[e] = {k: order.index(k) + 1 for k in variants}
    rows = [[k] + [ranks[e][k] for e in exts] for k in variants]
    from scipy.stats import spearmanr
    rhos = [spearmanr([ranks[a][k] for k in variants], [ranks[b][k] for k in variants])[0]
            for i, a in enumerate(exts) for b in exts[i + 1:]]
    return table(["Variant"] + exts, rows), (float(np.mean(rhos)), float(np.min(rhos))) if rhos else (float("nan"),) * 2


def order_block(res, exts):
    L = ["## Order of the variants according to the extractor (1 = best; without Seabed)", "",
         "Each table orders the variants by the indicated metric with each extractor. The mean and minimum Spearman "
         "between pairs of extractors is given (1 = same order). Variants whose differences fall within the CI should "
         "not be read as ordered.", ""]
    for m, title in (("auc", "Grouped AUC"), ("kid", "KID"), ("cmmd", "CMMD"), ("fd", "FD"), ("tstr_bacc", "TSTR (bacc)")):
        t, (mean, mn) = rank_section(res, m, exts)
        L += [f"### {title}: mean Spearman between extractors {f(mean, 2)} (minimum {f(mn, 2)})", ""] + t
    return L


def extractor_block(res, exts):
    L = ["## Panel per extractor (real reference = TRAIN, and VAL separately)", ""]
    for e in exts:
        sc = res["main"][e]
        t1, t2 = main_tables(e, sc)
        L += [f"### {e}: TRAIN reference", ""] + t1 + [f"### {e}: VAL reference (no shared TRAIN crops)", ""] + t2
        L += [f"### {e}: per-class AUC (out-of-fold scores of the global classifier)", ""] + class_table(sc, "auc")
        L += [f"### {e}: TSTR recall per class (train on pasted crops, evaluate on real VAL)", ""] + class_table(sc, "tstr_recall")
        L += [f"### {e}: coverage per class (VAL reference)", ""] + class_table(sc, "coverage@val")
    return L


def seven_block(res, exts):
    L = ["## Same set of classes: 7 classes and common minimum quota (to include Synthetic Seabed)", "",
         f"Quotas: {res['_meta']['quotas_7classes']} (Mask = 0).", ""]
    for e in exts:
        sc = res["seven_classes"][e]
        ks = ["_floor"] + names(sc)
        L += [f"### {e}", ""] + table(
            ["Variant", "Grouped AUC", "KID", "CMMD", "Coverage@val", "Density@val", "TSTR bacc"],
            [[NAMES.get(k, k), f"{f(sc[k]['global']['auc'])} ± {f(sc[k]['global']['auc_std'])}", f(sc[k]["global"]["kid"], 4),
              f(sc[k]["global"]["cmmd"], 3), f(sc[k]["global"]["coverage@val"], 2), f(sc[k]["global"]["density@val"], 2),
              f(sc[k]["global"]["tstr_bacc"], 3)] for k in ks])
    return L


def validation_block(res, val, exts):
    L = ["## Validation against the AP of the transfer check", "", f"**Warning:** {val['warning']}", ""]
    L += [f"Arms used: {val['exps']}. Warnings: {val['warnings'] or 'none'}.", ""]
    L += table(["Arm", "Global AP"] + [f"AP {c}" for c in CLASSES],
               [[e, f(val["ap"][e]["AP"], 2)] + [f(val["ap"][e].get(f"AP_{c}"), 2) for c in CLASSES] for e in val["exps"]])
    for e in exts:
        v = val["per_extractor"][e]
        rows = [[m, ", ".join(f"{k}={f(x, 3)}" for k, x in a["values"].items()), f"{a['pairs'][0]}/{a['pairs'][1]}",
                 f"{a['pairs_dAP'][0]}/{a['pairs_dAP'][1]}", f(a["spearman"], 2)] for m, a in v["global"].items()]
        L += [f"### {e}: global agreement (global AP)", ""] + table(
            ["Metric", "Value per arm", "Concordant pairs", "Pairs with |dAP| ≥ 1", "Spearman"], rows)
        rows = [[m, f"{a['pairs_pooled'][0]}/{a['pairs_pooled'][1]}",
                 f"{a['pairs_pooled_dAP'][0]}/{a['pairs_pooled_dAP'][1]}",
                 f(np.nanmean([c["spearman"] for c in a["per_class"].values()] or [np.nan]), 2)]
                for m, a in v["per_class"].items()]
        L += [f"### {e}: per-class agreement (8 classes × batches; the pairs of one class are not independent)", ""] + table(
            ["Metric", "Concordant pairs (all classes)", "Pairs with |dAP class| ≥ 1", "Mean Spearman per class"], rows)
    return L


def write(path=None):
    res = json.load(open(OUT / "gap_panel.json", encoding="utf8"))
    exts = list(res["main"])
    vf, bf = OUT / "gap_panel_validation.json", OUT / "gap_panel_failed_extractors.json"
    val = json.load(open(vf, encoding="utf8")) if vf.exists() else None
    bad = json.load(open(bf, encoding="utf8")) if bf.exists() else {}
    L = ["# Real-vs-synthetic gap metrics panel (CPU)", ""]
    summary = OUT / "gap_panel_summary.md"
    if summary.exists():
        L += [summary.read_text(encoding="utf8"), ""]
    L += ["## Extractors and licenses", ""] + table(
        ["Extractor", "Weights", "License", "Dim", "Notes"],
        [[e, m["hf"], m["license"], m["dim"], m["notes"]] for e, m in res["_meta"]["extractors"].items()])
    L += [f"Unavailable: {bad or 'none'}.", "",
          f"Quotas per class (variant_pilot): {res['_meta']['quotas_v11']}. Quotas of the transfer-check batches: "
          f"{res['_meta']['quotas_transfer']}.", ""]
    L += order_block(res, exts) + extractor_block(res, exts) + seven_block(res, exts)
    if val:
        L += validation_block(res, val, exts)
    Path(path or OUT / "gap_panel.md").write_text("\n".join(L) + "\n", encoding="utf8")
