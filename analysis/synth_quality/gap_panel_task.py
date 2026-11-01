"""Task evaluator: the gap panel with the features of an RT-DETRv2 trained on real data only.

Extractor `rtdetrv2_real10` (see gap_extractors): `px_real10/best.pth` (R50, 8 classes, 10 % real TRAIN), on CPU from the
RT-DETRv2 repository (no container). Same crop with context (box +20 % per side) squashed to 640x640 as in VAL; the
vector of an object = mean of the P3 map (stride 8, 256 d) of the HybridEncoder inside the box. Sensitivity (same
pass): P5 of the encoder and C5 of the backbone. Own cache: RUNS_DIR/synth_quality/cache/gap_panel_task/ (it touches
neither cache/gap_panel nor gap_panel.json).

Usage (CPU only, idle priority, a single process; every stage is resumable; requires the stages of gap_panel.py and
gap_panel_drop.py first):
  python analysis/synth_quality/gap_panel_task.py features   # panel crops + transfer-check batches + s_drop/control batch
  python analysis/synth_quality/gap_panel_task.py analyze    # panel + validation against the AP + s_drop verdict
  python analysis/synth_quality/gap_panel_task.py report     # out/gap_panel_task.{json,md}
TRAIN and VAL of the group partition only; never TEST.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_extractors as gx  # noqa: E402
import gap_panel as gp  # noqa: E402
import gap_panel_core as gc  # noqa: E402
import gap_panel_drop as gd  # noqa: E402
import gap_panel_validate as gv  # noqa: E402

OUT, dg, av = gp.OUT, gp.dg, gp.av
CACHE_TASK = gx.CACHE_MAIN.parent / "gap_panel_task"
MAIN = gx.TASK[0]
EXTS = gx.TASK + gx.TASK_SENS
OTHERS = gx.ORDER
FILES = {"json": OUT / "gap_panel_task.json", "md": OUT / "gap_panel_task.md"}


def use_cache():
    gx.CACHE = CACHE_TASK


# ------------------------------------------------------------------ stages
def stage_features(exts=EXTS):
    use_cache()
    gp.stage_features(exts)
    items, _ = gd.build(gd.ROOT)
    for n, it in items.items():
        gx.extract_set(n, it, exts)
        print(f"== {n}: {len(it)} crops", flush=True)


def drop_run(exts):
    """s_drop vs p_pos_tight with each extractor of `exts`; reuses gap_panel_drop (same rule and quotas)."""
    items, quotas = gd.build(gd.ROOT)
    out = {"quotas": quotas, "per_extractor": {}}
    for ext in exts:
        packs = {n: gp.load_pack(n, it, ext) for n, (it, _) in gp.real_sets().items() if n.startswith("real_")}
        for n, it in items.items():
            packs[n] = gp.load_pack(n, it, ext, flag=gd.CONTROL)
        out["per_extractor"][ext] = gp.panel_scenario(ext, list(items), quotas, packs)
    return out


def drop_verdicts(task_drop):
    """Verdict with the task evaluator alone and with it as a sixth extractor next to the 5 of the panel."""
    old = json.load(open(OUT / "gap_panel_drop.json", encoding="utf8"))["per_extractor"]
    pick = lambda sc: {"control": sc[f"drop_{gd.CONTROL}"]["global"], "test": sc[f"drop_{gd.TEST}"]["global"]}  # noqa: E731
    res5 = {e: pick(old[e]) for e in OTHERS}
    res_t = {e: pick(task_drop["per_extractor"][e]) for e in EXTS}
    return {"task_only": gd.verdict({MAIN: res_t[MAIN]}), "five": gd.verdict(res5),
            "six": gd.verdict({**res5, MAIN: res_t[MAIN]}), "sensitivity": {e: gd.verdict({e: res_t[e]}) for e in EXTS},
            "globals": {e: res_t[e] for e in EXTS}}


def stage_analyze(exts=EXTS):
    use_cache()
    sets, q4 = gp.all_sets()
    plan = json.load(open(av.ACACHE / "plan.json"))["quotas"]
    seabed_q = json.load(open(dg.CACHE / "external_seabed" / "plan.json"))["quotas"]
    q7 = {c: min(plan[c], seabed_q[c]) for c in dg.CLASSES}
    res = {"_meta": gp.meta_block(exts, plan, q4, q7, gp.VARIANTS)}
    for ext in exts:
        main, seven, transfer = gp.scenarios(ext, sets, plan, q7, q4, gp.VARIANTS)
        res.setdefault("main", {})[ext] = main
        res.setdefault("seven_classes", {})[ext] = seven
        res.setdefault("transfer", {})[ext] = transfer
    res["validation"] = validate(res["transfer"])
    res["drop"] = drop_verdicts(drop_run(exts))
    json.dump(res, open(FILES["json"], "w", encoding="utf8"), indent=1, default=float)
    return res


def validate(transfer):
    """Agreement with the AP in the same arms (and AP) as the panel validation (`gap_panel_validation.json`)."""
    base = json.load(open(OUT / "gap_panel_validation.json", encoding="utf8"))
    return {"exps": base["exps"], "per_extractor": {e: gv.validate_ext(r, base["ap"], gp.LOTS, base["exps"])
                                                    for e, r in transfer.items()}}


# ------------------------------------------------------------------ pure analyses (testable)
def rank_of(values, sense):
    """{key: rank} (1 = best synthetic quality)."""
    order = gc.rank_order(values, sense)
    return {k: order.index(k) + 1 for k in values}


def spearman_ranks(a, b):
    from scipy.stats import spearmanr
    keys = sorted(set(a) & set(b))
    return float(spearmanr([a[k] for k in keys], [b[k] for k in keys])[0]) if len(keys) > 2 else float("nan")


def ci_overlap_pairs(auc, ci):
    """(pairs with disjoint CI, total pairs) among variants: {v: auc}, {v: [lo, hi]}."""
    keys = sorted(auc)
    tot = sep = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            tot += 1
            sep += int(ci[a][1] < ci[b][0] or ci[b][1] < ci[a][0])
    return sep, tot


def spread_over_noise(auc, auc_std, ci):
    """Dispersion (standard deviation) of the AUC across variants divided by the mean half-width of their CI."""
    half = np.mean([(ci[k][1] - ci[k][0]) / 2 for k in auc])
    return float(np.std(list(auc.values())) / half) if half > 0 else float("nan")


def variants_of(sc):
    return [k for k in sc if not k.startswith("_") and k != gp.SEABED and k != gp.REF]


def metric_by(sc, metric, keys):
    return {k: sc[k]["global"][metric] for k in keys}


def discrimination(sc, keys=None):
    """Summary of the discrimination among the variants of a panel scenario for one extractor."""
    keys = keys or variants_of(sc)
    auc = metric_by(sc, "auc", keys)
    ci = {k: sc[k]["global"]["auc_ci"] for k in keys}
    sep, tot = ci_overlap_pairs(auc, ci)
    return {"auc": auc, "separated": sep, "pairs": tot,
            "spread_over_noise": spread_over_noise(auc, None, ci)}


def separation(sc, a, b):
    """AUC difference b - a with the combined CI half-width: (dAUC, combined noise, disjoint CIs?)."""
    ga, gb = sc[a]["global"], sc[b]["global"]
    ha, hb = (ga["auc_ci"][1] - ga["auc_ci"][0]) / 2, (gb["auc_ci"][1] - gb["auc_ci"][0]) / 2
    d = gb["auc"] - ga["auc"]
    disj = ga["auc_ci"][1] < gb["auc_ci"][0] or gb["auc_ci"][1] < ga["auc_ci"][0]
    return {"dAUC": d, "noise": float(np.hypot(ha, hb)), "disjoint": bool(disj)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["features", "analyze", "report"])
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    os.environ["GAP_THREADS"] = a.threads
    dg.lower_priority()
    if a.stage == "features":
        stage_features()
    elif a.stage == "analyze":
        stage_analyze()
    else:
        import gap_panel_task_report as tr
        tr.write()


if __name__ == "__main__":
    main()
