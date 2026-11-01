"""Adoption rule: variant `s_drop` against its control `p_pos_tight` (same batch, 300 images).

Hygiene change (it only removes anomalous crops from the library): ADOPT if the panel does not get worse beyond noise
in 3 or more of the 5 extractors and the diversity (Vendi and coverage) does not drop. Operationalization (set before
looking at the numbers):
  * "worse" in an extractor if grouped_AUC(s_drop) - grouped_AUC(control) > 0.02 or AUC(s_drop) exceeds the upper limit
    of the CI (group bootstrap) of the control.
  * "diversity drops" in an extractor if the mean Vendi of s_drop < 0.95 x that of the control, or the coverage with
    VAL reference < coverage of the control - 0.05 (VAL: no shared TRAIN crops). Diversity does not drop if it does not drop
    in >= 3 of the 5 extractors.
Separate cache: RUNS_DIR/synth_quality/cache/gap_panel_drop/ (the real caches are read from .../gap_panel/).
Requires `variant_pilot.py --stage features` and `gap_panel.py features` first (real features, quotas).

Usage: python analysis/synth_quality/gap_panel_drop.py [--root PILOT_ROOT]
Output: out/gap_panel_drop.json and out/gap_panel_drop.md.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_extractors as gx  # noqa: E402
import gap_panel as gp  # noqa: E402

av, dg, OUT = gp.av, gp.dg, gp.OUT
ROOT = av.PILOT_ROOT
CONTROL, TEST = "p_pos_tight", "s_drop"
CACHE_DROP = gx.CACHE_MAIN.parent / "gap_panel_drop"
AUC_TOL, VENDI_TOL, COV_TOL, MIN_OK = 0.02, 0.95, 0.05, 3
RULE = ("Rule (set before looking at the numbers):\n"
        "  * \"worse\" in an extractor if grouped_AUC(s_drop) - grouped_AUC(control) > 0.02 or AUC(s_drop) exceeds the "
        "upper limit\n    of the CI (group bootstrap) of the control.\n"
        "  * \"diversity drops\" in an extractor if the mean Vendi of s_drop < 0.95 x that of the control, or the "
        "coverage with VAL reference\n    < coverage of the control - 0.05 (VAL: no shared TRAIN crops). Diversity does not "
        "drop if it does not drop in >= 3 of the 5 extractors.")


def verdict(res):
    """res: {extractor: {'control': global, 'test': global}} -> verdict according to the rule (pure, testable)."""
    rows, worse, div_drop = {}, 0, 0
    for e, d in res.items():
        c, t = d["control"], d["test"]
        w = (t["auc"] - c["auc"] > AUC_TOL) or (t["auc"] > c["auc_ci"][1])
        dd = (t["vendi"] < VENDI_TOL * c["vendi"]) or (t["coverage@val"] < c["coverage@val"] - COV_TOL)
        rows[e] = {"dAUC": t["auc"] - c["auc"], "worse": bool(w), "diversity_drop": bool(dd)}
        worse += w
        div_drop += dd
    n = len(res)
    no_worse, no_drop = n - worse, n - div_drop
    ok = no_worse >= MIN_OK and no_drop >= MIN_OK
    return {"adopt": bool(ok), "extractors_not_worse": no_worse, "extractors_not_less_diverse": no_drop,
            "n_extractors": n, "per_extractor": rows}


def build(root):
    names = [CONTROL, TEST]
    samp, quotas = av.sample_variants(root, names, 300)       # quotas common to both and to real TRAIN (writes nothing)
    items = {f"drop_{n}": samp[n]["obj"] for n in names}
    return items, quotas


def run(root, exts):
    items, quotas = build(root)
    out = {"quotas": quotas, "per_extractor": {}}
    for ext in exts:
        packs = {n: gp.load_pack(n, it, ext) for n, (it, _) in gp.real_sets().items() if n.startswith("real_")}
        with gx.cache_at(CACHE_DROP):
            for n, it in items.items():
                gx.extract_set(n, it, [ext])
            for n, it in items.items():
                packs[n] = gp.load_pack(n, it, ext, flag=CONTROL)
        out["per_extractor"][ext] = gp.panel_scenario(ext, list(items), quotas, packs)
    res = {e: {"control": r[f"drop_{CONTROL}"]["global"], "test": r[f"drop_{TEST}"]["global"]}
           for e, r in out["per_extractor"].items()}
    out["verdict"] = verdict(res)
    json.dump(out, open(OUT / "gap_panel_drop.json", "w", encoding="utf8"), indent=1, default=float)
    write_md(out, res)
    return out


def write_md(out, res):
    v = out["verdict"]
    L = ["# s_drop vs p_pos_tight (adoption rule)", "",
         f"**Verdict: {'ADOPT' if v['adopt'] else 'DO NOT ADOPT'}.** Not worse: {v['extractors_not_worse']}/"
         f"{v['n_extractors']}; diversity not lower: {v['extractors_not_less_diverse']}/{v['n_extractors']} "
         f"(required minimum {MIN_OK}). Common quotas: {out['quotas']}.", "",
         "| Extractor | AUC control [CI] | AUC s_drop [CI] | dAUC | worse | Vendi c / s | Cov@val c / s | Cov (TRAIN) c / s "
         "| KID c / s | CMMD c / s | diversity drop |", "|" + "---|" * 11]
    for e, d in res.items():
        c, t, r = d["control"], d["test"], v["per_extractor"][e]
        L.append(f"| {e} | {c['auc']:.3f} [{c['auc_ci'][0]:.3f}; {c['auc_ci'][1]:.3f}] | {t['auc']:.3f} "
                 f"[{t['auc_ci'][0]:.3f}; {t['auc_ci'][1]:.3f}] | {r['dAUC']:+.3f} | {'yes' if r['worse'] else 'no'} | "
                 f"{c['vendi']:.1f} / {t['vendi']:.1f} | {c['coverage@val']:.2f} / {t['coverage@val']:.2f} | "
                 f"{c['coverage']:.2f} / {t['coverage']:.2f} | {c['kid']:.4f} / {t['kid']:.4f} | {c['cmmd']:.3f} / "
                 f"{t['cmmd']:.3f} | {'yes' if r['diversity_drop'] else 'no'} |")
    L += ["", RULE]
    Path(OUT / "gap_panel_drop.md").write_text("\n".join(L) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--extractors", nargs="*", default=None)
    a = ap.parse_args()
    dg.lower_priority()
    os.environ.setdefault("GAP_THREADS", "8")
    exts, bad = gx.available(a.extractors or gx.ORDER)
    print("extractors:", exts, bad, flush=True)
    run(a.root, exts)


if __name__ == "__main__":
    main()
