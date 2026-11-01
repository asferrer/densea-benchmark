"""Validation of the panel against the AP on VAL of the transfer check (global and per class). No GPU.

AP: a JSON table of the transfer-check arms (`--ap-table`, default analysis/proxy/out/px_val.json when it exists;
`{"runs": {arm: {"AP": ..., "AP_<class>": ...}}}`) and, for arms that are not in the table (e.g. `px_v9_real10_s2`), recomputed from
`paths.predictions_dir("rtdetrv2", "val")/<exp>_val_preds.json` with `eval.coco_tools.official` on the VAL annotations.
Each arm is one observation; the arms of the same batch (seeds) share the panel metrics. A batch = one folder of
synthetic images (LOTS in gap_panel.py).

Adding a second seed (once its predictions exist):
  python analysis/synth_quality/gap_panel.py validate --exps px_v9_real10_s2 px_p_real10_s2
  python analysis/synth_quality/gap_panel.py report
Adding another batch: a new entry in LOTS (gap_panel.py) and `gap_panel.py features` + `analyze` + `validate`.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import gap_panel_core as gc  # noqa: E402
from common import paths  # noqa: E402

OUT = HERE / "out"
AP_CACHE = paths.RUNS_DIR / "synth_quality" / "gap_panel_ap.json"
DEFAULT_AP_TABLE = paths.KIT / "analysis" / "proxy" / "out" / "px_val.json"
CLASSES = paths.CLASSES
GLOBAL_METRICS = ["auc", "fd", "kid", "cmmd", "cmmd_unb", "precision", "recall", "density", "coverage",
                  "fd@val", "kid@val", "cmmd@val", "precision@val", "recall@val", "density@val", "coverage@val",
                  "nn_real", "vendi", "nn_dist", "tstr_bacc"]
CLASS_METRICS = ["auc", "kid", "cmmd", "precision", "recall", "density", "coverage", "kid@val", "cmmd@val",
                 "precision@val", "recall@val", "density@val", "coverage@val", "vendi", "tstr_recall"]
MIN_DAP = 1.0     # threshold of a "noticeable AP difference" (AP points) for the strict variant of the agreement


def lot_of(exp, lots):
    """Batch an arm belongs to (drops the seed suffix `_s2`)."""
    base = re.sub(r"_s\d+$", "", exp)
    for k, spec in lots.items():
        if spec["exp"] == base:
            return k
    raise KeyError(f"arm without a known batch: {exp}")


def preds_file(exp):
    return paths.predictions_dir("rtdetrv2", "val") / f"{exp}_val_preds.json"


def recompute_ap(exp):
    """Global and per-class AP (x100) of an arm from its predictions on VAL."""
    from eval import coco_tools as ct
    gt = ct.load_gt(paths.annotations_path("val"))
    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    dt = ct.load_dets(preds_file(exp), gt)
    r = ct.official(gt, dt, gt.getImgIds(), names)
    return {k: round(100 * v, 2) for k, v in r.items()}


def ap_table(exps, lots, table_file=None, force=False):
    """AP per arm: from the AP table, from the cache or recomputed. Returns ({exp: {AP, AP_<class>...}}, notes)."""
    table_file = table_file or (DEFAULT_AP_TABLE if DEFAULT_AP_TABLE.exists() else None)
    ap = dict(json.load(open(table_file, encoding="utf8"))["runs"]) if table_file else {}
    cache = json.load(open(AP_CACHE, encoding="utf8")) if AP_CACHE.exists() else {}
    notes = []
    for e in exps:
        if e in ap:
            continue
        if e in cache and not force:
            ap[e] = cache[e]
        elif preds_file(e).exists():
            ap[e] = cache[e] = recompute_ap(e)
        else:
            notes.append(f"{e}: {e}_val_preds.json not found; skipped")
    AP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(AP_CACHE, "w", encoding="utf8"), indent=1)
    keep = {e: v for e, v in ap.items() if e in set(exps) | {l["exp"] for l in lots.values()}}
    return keep, notes


def _vals(ext_res, metric, lots, exps, cls=None):
    """{arm: value of the metric of the arm's batch} (global or of one class)."""
    out = {}
    for e in exps:
        lot = lot_of(e, lots)
        row = ext_res[lot]["global"] if cls is None else ext_res[lot]["per_class"].get(cls, {})
        v = row.get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[e] = v
    return out


def agreement(vals, ap, metric):
    s = gc.sense_of(metric)
    return {"values": {k: vals[k] for k in sorted(vals)}, "order": gc.rank_order(vals, s),
            "pairs": list(gc.pairwise_agreement(vals, ap, s)),
            "pairs_dAP": list(gc.pairwise_agreement(vals, ap, s, MIN_DAP)), "spearman": gc.spearman(vals, ap, s)}


def _class_agreement(ext_res, ap, lots, exps, metric):
    per, ok, tot, ok1, tot1 = {}, 0, 0, 0, 0
    for c in CLASSES:
        vals = _vals(ext_res, metric, lots, exps, c)
        apc = {e: ap[e][f"AP_{c}"] for e in vals}
        if len(vals) < 2:
            continue
        a = agreement(vals, apc, metric)
        per[c] = {"spearman": a["spearman"], "pairs": a["pairs"], "pairs_dAP": a["pairs_dAP"], "values": a["values"],
                  "ap": apc}
        ok, tot = ok + a["pairs"][0], tot + a["pairs"][1]
        ok1, tot1 = ok1 + a["pairs_dAP"][0], tot1 + a["pairs_dAP"][1]
    return {"per_class": per, "pairs_pooled": [ok, tot], "pairs_pooled_dAP": [ok1, tot1]}


def validate_ext(ext_res, ap, lots, exps):
    """Agreement with the AP of each metric: global and per class (with pairs pooled over classes)."""
    g = {m: agreement(_vals(ext_res, m, lots, exps), {e: ap[e]["AP"] for e in exps}, m) for m in GLOBAL_METRICS}
    pc = {m: _class_agreement(ext_res, ap, lots, exps, m) for m in CLASS_METRICS}
    return {"global": g, "per_class": pc}


def run(extra_exps, force=False, table_file=None):
    res = json.load(open(OUT / "gap_panel.json", encoding="utf8"))
    import gap_panel as gp
    lots = gp.LOTS
    exps = [spec["exp"] for spec in lots.values()] + list(extra_exps)
    ap, notes = ap_table(exps, lots, table_file, force)
    exps = [e for e in exps if e in ap]
    n_batches = len({lot_of(e, lots) for e in exps})
    out = {"exps": exps, "ap": ap, "batches": n_batches, "warnings": notes,
           "spearman_p_min": {str(n): gc.spearman_min_p(n) for n in range(3, 8)},
           "warning": f"{n_batches} distinct batches and {len(exps)} arms: with so few points there is no reliable "
                      f"correlation (exact minimum Spearman p with n = {n_batches}: {gc.spearman_min_p(n_batches):.3f}); "
                      "the pairs are descriptive. With a single seed, the AP difference between batches may be noise.",
           "per_extractor": {e: validate_ext(r, ap, lots, exps) for e, r in res["transfer"].items()}}
    json.dump(out, open(OUT / "gap_panel_validation.json", "w", encoding="utf8"), indent=1, default=float)
    print(out["warning"], notes)
    return out
