"""Real-vs-synthetic gap panel (CPU only): grouped AUC, KID, CMMD, FD, precision/recall, density/coverage, Vendi, NN and
crop-level TSTR, for several extractors (see gap_extractors.REGISTRY) and on the SAME crops used by `variant_pilot`
(same per-class quotas and seeds: the `*_meta.json` files of the variant_pilot cache are read).

Usage (CPU only, idle priority, a single process; requires `variant_pilot.py --stage features` and
`external_seabed.py --stage features` first):
  python analysis/synth_quality/gap_panel.py features [--extractors resnet50 dinov2_b14 ...]
  python analysis/synth_quality/gap_panel.py analyze          # panel per variant, TSTR and transfer-check batches
  python analysis/synth_quality/gap_panel.py validate [--exps px_v9_real10_s2 px_p_real10_s2]
  python analysis/synth_quality/gap_panel.py report           # out/gap_panel.{md,json}
TRAIN and VAL of the group partition only; never TEST. It does not rewrite the plan.json of variant_pilot (it only
reads it).
Real reference = TRAIN crops (like the AUC of variant_pilot) and, separately, VAL crops ("@val"): the pasted objects of
copy-paste come from TRAIN crops, so with a TRAIN reference the fidelity could be inflated (the distance to
the closest real item, `nn_real`, shows near-exact copies); VAL does not have that problem.

Transfer-check batches (LOTS): the exact synthetic sets of the transfer check (RT-DETRv2 trained on 10 % real plus a
synthetic set): px_v9 (clean-background batch, read from its training JSON in PREPARE_OUT) and the copy-paste pilot
folders p_pos_tight and q_lab_same under TRANSFER_SYNTH (default DATA_ROOT/synthetic/transfer_check).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_extractors as gx  # noqa: E402
import gap_panel_core as gc  # noqa: E402
import variant_pilot_core as av  # noqa: E402
from common import paths  # noqa: E402

dg = av.dg
OUT = dg.OUT
VARIANTS = ["c_v10_noaug", "g_v11_copypaste", "j_v11_visible", "m_pos", "p_pos_tight", "q_base", "q_lab", "q_lab_same",
            "q_lab_local", "r_matte", "r_medium_harm"]
REF, SEABED = av.BASELINE, "external_seabed"
TRANSFER_SYNTH = Path(os.environ.get("TRANSFER_SYNTH", paths.SYNTH_DIR / "transfer_check"))
# Exact transfer-check batches: id -> (COCO folder | train.json, variant name for the group criterion, AP arm)
LOTS = {"px_v9": {"train_json": str(paths.PREPARE_OUT / "px_v9_real10" / "train.json"), "flag": "px_v9",
                  "exp": "px_v9_real10"},
        "px_p": {"coco": str(TRANSFER_SYNTH / "p_pos_tight"), "flag": "p_pos_tight", "exp": "px_p_real10"},
        "px_q": {"coco": str(TRANSFER_SYNTH / "q_lab_same"), "flag": "q_lab_same", "exp": "px_q_real10"}}
CAP = 125
MIN_N = 12          # minimum n per side for per-class metrics
THREADS_DEFAULT = "8"


# ------------------------------------------------------------------ crop sets (the same as variant_pilot)
def real_sets():
    s = json.load(open(dg.WORK / "samples.json", encoding="utf8"))
    return {"real_train_obj": (s["real_train_obj"], "real_train_obj"), "real_val_obj": (s["real_val_obj"], "real_val_obj"),
            REF: (s["synth_obj"], "synth_obj")}


def variant_sets(names):
    out = {}
    for n in names:
        out[n] = (json.load(open(av.ACACHE / f"{n}_meta.json", encoding="utf8"))["obj"], n)
    return out


def seabed_set():
    f = dg.CACHE / "external_seabed" / "external_seabed_meta.json"
    return {SEABED: (json.load(open(f, encoding="utf8"))["obj"], SEABED)}


def lot_items(spec, quotas):
    """Crops of a transfer-check batch with per-class quota; pasted annotations only."""
    if "train_json" in spec:
        d = json.load(open(spec["train_json"], encoding="utf8"))
        imgs = {i["id"]: {"id": i["id"], "path": str(dg.DATA_ROOT / i["file_name"]), "source": "synthetic",
                          "w": i["width"], "h": i["height"]} for i in d["images"] if i.get("source") == "synthetic"}
        anns = [{"img": a["image_id"], "bbox": a["bbox"], "cls": dg.CLASSES[a["category_id"]]}
                for a in d["annotations"] if a["image_id"] in imgs and not a.get("iscrowd", 0)]
    else:
        imgs, anns = av.load_variant(spec["coco"])
    return dg.sample_objects(imgs, anns, quotas)[0], dg.sample_objects(imgs, anns, {c: 10 ** 9 for c in dg.CLASSES})[1]


def lot_quotas():
    """Common quota of the transfer-check batches: minimum over batches and real TRAIN (capped at CAP)."""
    s = real_sets()["real_train_obj"][0]
    have = [{c: sum(m["cls"] == c for m in s) for c in dg.CLASSES}]
    for spec in LOTS.values():
        have.append(lot_items(spec, {c: 0 for c in dg.CLASSES})[1])
    return av.common_quota(have, CAP)


def all_sets(variants=VARIANTS):
    sets = {**real_sets(), **variant_sets(variants), **seabed_set()}
    q4 = lot_quotas()
    for k, spec in LOTS.items():
        sets[k] = (lot_items(spec, q4)[0], None)
    return sets, q4


def stage_features(exts, variants=VARIANTS):
    """Extract and cache the features of every set for the requested extractors (resumable)."""
    sets, _ = all_sets(variants)
    for name, (items, shared) in sets.items():
        gx.extract_set(name, items, exts, shared_name=shared)
        print(f"== {name}: {len(items)} crops", flush=True)


# ------------------------------------------------------------------ packs with groups
def load_pack(name, items, ext, flag=None):
    """{'f': features, 'meta': items, 'g': CV groups} of a set for one extractor."""
    f = np.load(gx.CACHE / ext / f"{name}.npy")
    if name.startswith("real_"):
        g = av.real_groups(items, "img")
    else:
        v = av.attach_groups({"img_meta": [], "obj_meta": items}, flag or name, av.background_group_map())
        g = v["obj_g"]
    return {"f": f, "meta": items, "g": list(g), "cls": [m["cls"] for m in items]}


def sub(pack, idx):
    idx = list(idx)
    return {"f": pack["f"][idx], "meta": [pack["meta"][i] for i in idx], "g": [pack["g"][i] for i in idx],
            "cls": [pack["cls"][i] for i in idx]}


def by_class(pack, c):
    return sub(pack, [i for i, k in enumerate(pack["cls"]) if k == c])


# ------------------------------------------------------------------ panel of a variant
def dist_block(real, fake, tag="", ci=False):
    """Real-vs-synthetic distribution metrics with suffix `tag` ('' = TRAIN ref, '@val' = VAL ref)."""
    m = gc.distribution_metrics(real["f"], fake["f"])
    out = {k + tag: v for k, v in m.items()}
    if ci and min(len(real["f"]), len(fake["f"])) >= 100:
        r = gc.group_subsample_ci(gc.distribution_metrics, real["f"], fake["f"], real["g"], fake["g"], b=30)
        out.update({k + tag + "_ci": list(v) for k, v in r.items()})
    return out


def class_block(real_tr, real_va, fake, classes):
    """Per-class metrics (no FD: with small n the covariance is singular)."""
    out = {}
    for c in classes:
        rt, rv, fk = by_class(real_tr, c), by_class(real_va, c), by_class(fake, c)
        if min(len(rt["f"]), len(fk["f"])) < MIN_N:
            continue
        row = {k: v for k, v in dist_block(rt, fk).items() if k != "fd"}
        if len(rv["f"]) >= MIN_N:
            row.update({k: v for k, v in dist_block(rv, fk, "@val").items() if k != "fd@val"})
        row["n"] = len(fk["f"])
        out[c] = row
    return out


def tstr_block(train, test, classes):
    r = gc.tstr(train["f"], train["cls"], test["f"], test["cls"], classes, test_groups=test["g"])
    return {"bacc": r["bacc"], "bacc_ci": list(r["bacc_ci"]), "recall": r["recall"], "n_train": r["n_train"]}


def panel_variant(real_tr, real_va, fake, quotas, test, ci=True):
    """Full panel of a variant against real TRAIN and VAL with the same per-class quotas."""
    classes = [c for c in dg.CLASSES if quotas.get(c, 0) >= MIN_N and any(k == c for k in fake["cls"])]
    auc = gc.grouped_auc(real_tr["f"], fake["f"], real_tr["g"], fake["g"], real_tr["cls"], fake["cls"])
    g = {"auc": auc["auc"], "auc_std": auc["auc_std"], "auc_ci": list(auc["auc_ci"]), "n_fake": auc["n_fake"],
         "n_real": auc["n_real"]}
    g.update(dist_block(real_tr, fake, "", ci))
    g.update({k: v for k, v in dist_block(real_va, fake, "@val").items()})
    div = av.diversity_by_class(fake["f"], fake["meta"], quotas)
    g.update({"vendi": div.get("mean", {}).get("vendi"), "nn_dist": div.get("mean", {}).get("nn_dist")})
    ts = tstr_block(fake, test, classes)       # same VAL set for every variant
    g.update({"tstr_bacc": ts["bacc"], "tstr_bacc_ci": ts["bacc_ci"]})
    pc = class_block(real_tr, real_va, fake, classes)
    for c, row in pc.items():
        row["auc"] = auc["per_class"].get(c)
        row["vendi"] = div.get(c, {}).get("vendi")
        row["nn_dist"] = div.get(c, {}).get("nn_dist")
        row["tstr_recall"] = ts["recall"].get(c)
    return {"global": g, "per_class": pc, "classes": classes}


def panel_floor(rt, rv, qv, ci=True):
    """Real floor: TRAIN (reference) against VAL (as "synthetic"); the @val block uses VAL as reference and TRAIN as
    "synthetic"; TSTR is TRTR (train on real TRAIN, evaluate on VAL)."""
    out = panel_variant(rt, rv, rv, qv, rv, ci)
    g, pc = out["global"], out["per_class"]
    g.update(dist_block(rv, rt, "@val"))
    ts = tstr_block(rt, rv, out["classes"])
    g.update({"tstr_bacc": ts["bacc"], "tstr_bacc_ci": ts["bacc_ci"], "trtr_bacc": ts["bacc"]})
    for c, row in pc.items():
        row["tstr_recall"] = ts["recall"].get(c)
        a, b = by_class(rv, c), by_class(rt, c)
        if min(len(a["f"]), len(b["f"])) >= MIN_N:
            row.update({k: v for k, v in dist_block(a, b, "@val").items() if k != "fd@val"})
    return out


def panel_scenario(ext, names, quotas, packs, ci=True, log=print):
    """Panel of every entry of `names` with common quotas `quotas` + real floor (TRAIN vs VAL) and TRTR."""
    tr, va = packs["real_train_obj"], packs["real_val_obj"]
    qv = {c: min(quotas[c], sum(k == c for k in va["cls"])) for c in dg.CLASSES}
    rt, rv = sub(tr, av.pick_by_class(tr["meta"], quotas)), sub(va, av.pick_by_class(va["meta"], qv))
    out = {"_quotas": quotas, "_quotas_val": qv}
    t0 = time.time()
    out["_floor"] = panel_floor(rt, rv, qv, ci)
    for n in names:
        p = packs[n]
        q = {c: min(quotas[c], sum(k == c for k in p["cls"])) for c in dg.CLASSES}
        f = sub(p, av.pick_by_class(p["meta"], q))
        rtn = sub(tr, av.pick_by_class(tr["meta"], q))
        rvn = sub(va, av.pick_by_class(va["meta"], {c: min(q[c], qv[c]) for c in q}))
        out[n] = panel_variant(rtn, rvn, f, q, rv, ci)
        out[n]["quotas"] = q
        log(f"  [{ext}] {n}: AUC {out[n]['global']['auc']:.3f}  ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ analyze stage
def meta_block(exts, plan, q4, q7, variants):
    return {"quotas_v11": plan, "quotas_transfer": q4, "quotas_7classes": q7, "variants": variants,
            "extractors": {e: {"hf": gx.REGISTRY[e].hf_id, "license": gx.REGISTRY[e].license,
                               "dim": gx.REGISTRY[e].dim, "notes": gx.REGISTRY[e].notes} for e in exts}}


def scenarios(ext, sets, plan, q7, q4, variants):
    """The three scenarios (main, seven classes, transfer check) of one extractor."""
    packs = {n: load_pack(n, it, ext, flag=LOTS.get(n, {}).get("flag")) for n, (it, _) in sets.items()}
    main = [REF] + variants + [SEABED]
    return (panel_scenario(ext, main, plan, packs), panel_scenario(ext, main, q7, packs),
            panel_scenario(ext, list(LOTS), q4, packs))


def stage_analyze(exts, variants=VARIANTS):
    sets, q4 = all_sets(variants)
    plan = json.load(open(av.ACACHE / "plan.json"))["quotas"]
    seabed_q = json.load(open(dg.CACHE / "external_seabed" / "plan.json"))["quotas"]
    q7 = {c: min(plan[c], seabed_q[c]) for c in dg.CLASSES}
    res = {"_meta": meta_block(exts, plan, q4, q7, variants)}
    for ext in exts:
        main, seven, transfer = scenarios(ext, sets, plan, q7, q4, variants)
        res.setdefault("main", {})[ext] = main
        res.setdefault("seven_classes", {})[ext] = seven
        res.setdefault("transfer", {})[ext] = transfer
    json.dump(res, open(OUT / "gap_panel.json", "w", encoding="utf8"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["features", "analyze", "validate", "report"])
    ap.add_argument("--extractors", nargs="*", default=None, help="default: those of gx.ORDER that are available")
    ap.add_argument("--exps", nargs="*", default=None, help="validate: extra arms, e.g. px_v9_real10_s2 px_p_real10_s2")
    ap.add_argument("--ap-table", default=None, help="validate: JSON with the AP of each arm ({'runs': {arm: {...}}}); "
                    "default analysis/proxy/out/px_val.json if it exists")
    ap.add_argument("--threads", default=THREADS_DEFAULT)
    a = ap.parse_args()
    os.environ["GAP_THREADS"] = a.threads
    dg.lower_priority()
    exts = a.extractors or gx.ORDER
    if a.stage in ("features", "analyze"):
        exts, bad = gx.available(exts)
        print("available extractors:", exts, "| unavailable:", bad, flush=True)
        json.dump(bad, open(OUT / "gap_panel_failed_extractors.json", "w", encoding="utf8"), indent=1)
    if a.stage == "features":
        stage_features(exts)
    elif a.stage == "analyze":
        stage_analyze(exts)
    elif a.stage == "validate":
        import gap_panel_validate as gv
        gv.run(a.exps or [], table_file=a.ap_table)
    else:
        import gap_panel_report as gr
        gr.write()


if __name__ == "__main__":
    main()
