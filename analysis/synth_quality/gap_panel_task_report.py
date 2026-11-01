"""Panel appendix: `out/gap_panel_task.md` (task evaluator, RT-DETRv2 with 10 % real) from `out/gap_panel_task.json` and
`out/gap_panel.json` (the 5 general extractors). The summary of the findings lives in `out/gap_panel_task_summary.md`
(when it exists it is placed first) so that regenerating the tables does not erase it."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_extractors as gx  # noqa: E402
import gap_panel_core as gc  # noqa: E402
import gap_panel_report as gr  # noqa: E402
import gap_panel_task as gt  # noqa: E402

OUT = HERE / "out"
MAIN, SENS = gt.MAIN, gx.TASK_SENS
ORDER6 = gx.ORDER + [MAIN]
f, table = gr.f, gr.table


def yes_no(flag):
    return "yes" if flag else "no"


def triple(s):
    """'dAUC / noise / disjoint' cell of a gt.separation result."""
    return f"{s['dAUC']:+.3f} / {f(s['noise'])} / {yes_no(s['disjoint'])}"


def load():
    base = json.load(open(OUT / "gap_panel.json", encoding="utf8"))
    task = json.load(open(gt.FILES["json"], encoding="utf8"))
    val = json.load(open(OUT / "gap_panel_validation.json", encoding="utf8"))
    sc = {k: {**base[k], **task[k]} for k in ("main", "seven_classes", "transfer")}
    return base, task, val, sc


def rank_tables(sc):
    """Rank of each variant per extractor and Spearman of the task evaluator with each one and with the mean rank of the 5."""
    variants = gt.variants_of(sc["main"][MAIN])
    L, summary = [], {}
    for m, title in (("auc", "Grouped AUC"), ("kid", "KID"), ("cmmd", "CMMD"), ("fd", "FD")):
        rk = {e: gt.rank_of(gt.metric_by(sc["main"][e], m, variants), gc.sense_of(m)) for e in ORDER6}
        mean5 = {k: float(np.mean([rk[e][k] for e in gx.ORDER])) for k in variants}
        rho = {e: gt.spearman_ranks(rk[MAIN], rk[e]) for e in gx.ORDER}
        rho5 = gt.spearman_ranks(rk[MAIN], mean5)
        inner = [gt.spearman_ranks(rk[a], rk[b]) for i, a in enumerate(gx.ORDER) for b in gx.ORDER[i + 1:]]
        summary[m] = {"rho": rho, "rho_mean5": rho5, "rho_among_the_5": float(np.mean(inner))}
        L += [f"### {title}: rank of each variant (1 = best synthetic quality)", ""]
        L += table(["Variant"] + ORDER6 + ["mean of the 5"], [[k] + [rk[e][k] for e in ORDER6] + [f(mean5[k], 1)]
                                                              for k in variants])
        L += [f"Spearman of `{MAIN}` with each extractor: " + ", ".join(f"{e} {f(r, 2)}" for e, r in rho.items()) +
              f". With the mean rank of the 5: {f(rho5, 2)}. Mean Spearman among the 5 general ones: "
              f"{f(summary[m]['rho_among_the_5'], 2)}.", ""]
    return L, summary


def v9_v11_block(sc):
    """Separation of v9 from v11: pilot batch (ref_batch_v9 vs p_pos_tight) and transfer-check batches (px_v9 vs px_p)."""
    L = ["Separation = AUC(v11) − AUC(v9). A lower AUC means less gap. \"Noise\" = square root of the sum of squares of "
         "the half-widths of the group CIs; \"disjoint\" = the CIs do not overlap.", ""]
    rows = []
    for e in ORDER6:
        a = gt.separation(sc["main"][e], "ref_batch_v9", "p_pos_tight")
        b = gt.separation(sc["transfer"][e], "px_v9", "px_p")
        c = gt.separation(sc["transfer"][e], "px_v9", "px_q")
        rows.append([e, f(sc["main"][e]["ref_batch_v9"]["global"]["auc"]), f(sc["main"][e]["p_pos_tight"]["global"]["auc"]),
                     triple(a), f(sc["transfer"][e]["px_v9"]["global"]["auc"]), f(sc["transfer"][e]["px_p"]["global"]["auc"]),
                     triple(b), triple(c)])
    return L + table(["Extractor", "AUC v9 (pilot)", "AUC p_pos_tight", "dAUC / noise / disjoint",
                      "AUC px_v9", "AUC px_p", "px_p − px_v9: dAUC / noise / disjoint", "px_q − px_v9"], rows)


def discrimination_block(sc):
    """Discrimination among the v11 variants and between p and q for each extractor."""
    rows, pq = [], []
    for e in ORDER6:
        d = gt.discrimination(sc["main"][e])
        rows.append([e, f"{d['separated']}/{d['pairs']}", f(d["spread_over_noise"], 2),
                     f(min(d["auc"].values())), f(max(d["auc"].values()))])
        pq.append([e, triple(gt.separation(sc["transfer"][e], "px_p", "px_q")),
                   triple(gt.separation(sc["main"][e], "p_pos_tight", "q_lab_same"))])
    L = ["Among the 11 variants (10 of v11 and c_v10_noaug) of the main scenario: pairs of variants with disjoint AUC CIs "
         "and dispersion of the AUC across variants divided by the mean CI half-width (the higher, the better it "
         "separates above its noise).", ""]
    L += table(["Extractor", "Pairs with disjoint CI", "Dispersion / CI half-width", "Minimum AUC", "Maximum AUC"], rows)
    L += ["p versus q (q − p): transfer-check batches (px_p, px_q) and pilot variants (p_pos_tight, q_lab_same).", ""]
    return L + table(["Extractor", "transfer check: dAUC / noise / disjoint", "pilot: dAUC / noise / disjoint"], pq)


def validation_block(val, task):
    """Agreement of the AUC, KID, CMMD and FD with the AP in the 5 arms (3 batches), task evaluator vs the 5 general ones."""
    ms = ["auc", "kid", "cmmd", "fd", "kid@val", "cmmd@val", "coverage@val", "recall@val", "tstr_bacc"]
    per = {e: val["per_extractor"][e]["global"] for e in gx.ORDER}
    per.update({e: task["validation"]["per_extractor"][e]["global"] for e in gt.EXTS})
    ap = val["ap"]
    L = [f"Arms: {val['exps']}; global AP: " + ", ".join(f"{e} {f(ap[e]['AP'], 2)}" for e in val["exps"]) +
         ". Concordant pairs (all / with |dAP| ≥ 1) and Spearman with the AP; only 3 batches (exact minimum p 0.33): "
         "descriptive.", ""]
    rows = [[e] + [f"{per[e][m]['pairs'][0]}/{per[e][m]['pairs'][1]} · {per[e][m]['pairs_dAP'][0]}/{per[e][m]['pairs_dAP'][1]} "
                   f"· {f(per[e][m]['spearman'], 2)}" for m in ms] for e in ORDER6 + SENS]
    L += table(["Extractor"] + ms, rows)
    L += ["Order of the batches according to the metric (best → worst), grouped AUC and CMMD:", ""]
    rows = [[e, " > ".join(per[e]["auc"]["order"]), " > ".join(per[e]["cmmd"]["order"])] for e in ORDER6 + SENS]
    return L + table(["Extractor", "AUC (arms)", "CMMD (arms)"], rows)


def drop_block(task):
    d = task["drop"]
    rows = []
    for e in gt.EXTS:
        c, t = d["globals"][e]["control"], d["globals"][e]["test"]
        r = d["sensitivity"][e]["per_extractor"][e]
        rows.append([e, f"{f(c['auc'])} [{f(c['auc_ci'][0])}; {f(c['auc_ci'][1])}]", f"{f(t['auc'])} [{f(t['auc_ci'][0])}; "
                     f"{f(t['auc_ci'][1])}]", f"{r['dAUC']:+.3f}", yes_no(r["worse"]),
                     f"{f(c['vendi'], 1)} / {f(t['vendi'], 1)}", f"{f(c['coverage@val'], 2)} / {f(t['coverage@val'], 2)}",
                     yes_no(r["diversity_drop"])])
    L = table(["Extractor", "AUC control [CI]", "AUC s_drop [CI]", "dAUC", "worse", "Vendi c / s", "Cov@val c / s",
               "low diversity"], rows)
    for k, t in (("five", "the 5 general ones"), ("six", "the 5 general ones + the task evaluator (6)")):
        v = d[k]
        L += [f"- Adoption rule with {t}: {'ADOPT' if v['adopt'] else 'DO NOT ADOPT'}; not worse "
              f"{v['extractors_not_worse']}/{v['n_extractors']}, diversity not lower "
              f"{v['extractors_not_less_diverse']}/{v['n_extractors']} (minimum 3 in each)."]
    v = d["task_only"]
    L += [f"- Task evaluator only: {'not worse' if v['extractors_not_worse'] else 'worse'} and diversity "
          f"{'not lower' if v['extractors_not_less_diverse'] else 'lower'}.", ""]
    return L


def sensitivity_block(sc):
    ks = gt.variants_of(sc["main"][MAIN])
    rows = [[k] + [f(sc["main"][e][k]["global"]["auc"]) for e in [MAIN] + SENS] for k in ["_floor", "ref_batch_v9"] + ks]
    rk = {e: gt.rank_of(gt.metric_by(sc["main"][e], "auc", ks), -1) for e in [MAIN] + SENS}
    L = table(["Variant"] + [MAIN, SENS[0], SENS[1]], rows)
    return L + [f"Spearman of ranks (AUC) P3–P5: {f(gt.spearman_ranks(rk[MAIN], rk[SENS[0]]), 2)}; P3–C5: "
                f"{f(gt.spearman_ranks(rk[MAIN], rk[SENS[1]]), 2)}.", ""]


def seven_block(sc):
    s7 = sc["seven_classes"][MAIN]
    ks = ["_floor"] + gr.names(s7)
    return table(["Variant", "Grouped AUC", "KID", "CMMD", "Coverage@val", "TSTR bacc"],
                 [[gr.NAMES.get(k, k), f(s7[k]["global"]["auc"]), f(s7[k]["global"]["kid"], 4), f(s7[k]["global"]["cmmd"], 3),
                   f(s7[k]["global"]["coverage@val"], 2), f(s7[k]["global"]["tstr_bacc"], 3)] for k in ks])


def write(path=None):
    base, task, val, sc = load()
    L = ["# Panel appendix: task evaluator (RT-DETRv2 with 10 % real)", ""]
    summary = OUT / "gap_panel_task_summary.md"
    if summary.exists():
        L += [summary.read_text(encoding="utf8"), ""]
    m = task["_meta"]["extractors"]
    L += ["## Extractors of the appendix", ""] + table(["Extractor", "Weights", "Dim", "Notes"],
                                                       [[e, x["hf"], x["dim"], x["notes"]] for e, x in m.items()])
    L += ["## Panel with the main extractor (TRAIN and VAL reference)", ""]
    t1, t2 = gr.main_tables(MAIN, sc["main"][MAIN])
    L += ["### TRAIN reference", ""] + t1 + ["### VAL reference", ""] + t2
    L += ["### Per-class AUC", ""] + gr.class_table(sc["main"][MAIN], "auc")
    L += ["### Transfer-check batches", ""]
    t1, t2 = gr.main_tables(MAIN, sc["transfer"][MAIN])
    L += t1 + t2
    rk, _ = rank_tables(sc)
    L += ["## Does it order the variants like the other extractors?", ""] + rk
    L += ["## Does it separate v9 from v11?", ""] + v9_v11_block(sc)
    L += ["## Does it discriminate better among the v11 variants and between p and q?", ""] + discrimination_block(sc)
    L += ["## Validation against the AP of the transfer check", ""] + validation_block(val, task)
    L += ["## s_drop vs p_pos_tight (adoption rule)", ""] + drop_block(task)
    L += ["## Sensitivity to the layer (grouped AUC)", ""] + sensitivity_block(sc)
    L += ["## Seven-class scenario (for Synthetic Seabed)", ""] + seven_block(sc)
    Path(path or gt.FILES["md"]).write_text("\n".join(L) + "\n", encoding="utf8")
