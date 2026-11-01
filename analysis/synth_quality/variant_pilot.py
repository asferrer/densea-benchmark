"""Metrics of the copy-paste pilot variants: realism and diversity of each variant (CPU only).

Uses `variant_pilot_core`. Variants (folders `<PILOT_ROOT>/<variant>/` with images/ and synthetic_dataset.json written
by the generator; v11 is the copy-paste batch label):
  c_v10_noaug, f_v11_rectfilter, g_v11_copypaste, h_v11_harmon, i_v11_all, j_v11_visible
  (j = i + no global veil, bounded blur and per-class visibility filter)
  k_sharp, l_scale, m_pos, n_vis15 (chained on j): k = j + single resampling and sharpness/grain matching;
  l = k + crop scale in [0.6, 1.5]; m = l + faithful position with a margin to the border; n = m + visibility
  threshold p15. Besides the AUC and the diversity, the report gives the sharpness (hf_ratio, lapvar64), the noise,
  the fraction of boxes at the border and the scale distribution of the pasted objects against the real ones
  (`variant_measures.py`).
  q_base, q_lab, q_lab_same, q_lab_local (all with a real TRAIN background, tight box and NO sharpness/grain
  matching): q_base = p_pos_tight without `match_sharpness`; q_lab = q_base + crop chosen by LAB similarity to the
  background (different partition group); q_lab_same = the same allowing the same group; q_lab_local = q_lab_same +
  position chosen by similarity of the ring to the ring of the crop at its origin.
  r_matte, r_matte_fg, r_medium, r_medium_harm (base q_lab_same): r_matte = + matting with trimap (closed-form alpha,
  F = original pixel); r_matte_fg = r_matte with the Germer foreground; r_medium = no LAB harmonization and
  ring-to-ring medium transfer (ring_gain lambda 0.5); r_medium_harm = medium transfer (lambda 1) before harmonization.
  s_clean: p_pos_tight with the clean crop library (mask clipped to the annotated box).
  s_drop: p_pos_tight with the library that only removes anomalous crops (original masks).

DIFFERENCE with the clean-background variants in the copy-paste variants (g, i, j, ...): the background is a REAL TRAIN
image whose real boxes are kept (COCO: `origin = "real"`). Those boxes are discarded here: the object crops of ALL
variants contain only the PASTED objects (`origin = "pasted"` or no `origin`). The per-IMAGE AUC of the copy-paste
variants drops trivially (the background is real) and is not comparable with c, f and h: the key metric is the
per-OBJECT AUC. With copy-paste the object crop includes the surrounding context (enlarged box), which is real.

Feature cache in RUNS_DIR/synth_quality/cache/variant_pilot/. Real items: the same as `variant_pilot_core` (domain_gap
cache, TRAIN and VAL only, never TEST). Requires `domain_gap.py --stage features` first.

Usage (CPU only, idle priority, a single process):
  python analysis/synth_quality/variant_pilot.py --stage features
  python analysis/synth_quality/variant_pilot.py --stage analyze
  python analysis/synth_quality/variant_pilot.py --stage all --baseline   # adds the v9 batch as a reference
Output: out/variant_pilot.{json,md}; mosaics in RUNS_DIR/synth_quality/mosaics/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variant_measures as measures  # noqa: E402
import variant_pilot_core as av  # noqa: E402

dg = av.dg
OUT, ACACHE = dg.OUT, av.ACACHE
REAL_BACKGROUND = av.REAL_BG_VARIANTS
VARIANTS = ["c_v10_noaug", "f_v11_rectfilter", "g_v11_copypaste", "h_v11_harmon", "i_v11_all", "j_v11_visible",
            "k_sharp", "l_scale", "m_pos", "n_vis15", "o_tight", "p_pos_tight",
            "q_base", "q_lab", "q_lab_same", "q_lab_local", "r_matte", "r_matte_fg", "r_medium", "r_medium_harm",
            "s_clean", "s_drop"]
load_variant, keep_pasted = av.load_variant, av.keep_pasted


def annotation_counts(root, names):
    """Per variant: pasted objects, kept real boxes and rectangular crops discarded per class."""
    out = {}
    for n in names:
        d = json.load(open(Path(root) / n / "synthetic_dataset.json", encoding="utf8"))
        origins = [a.get("origin", "pasted") for a in d["annotations"] if not a.get("iscrowd", 0)]
        gen = d.get("info", {}).get("synth_generator", {})
        rect, vis = gen.get("rect_filter") or {}, gen.get("visibility_filter") or {}
        out[n] = {"pasted": origins.count("pasted"), "kept_real": origins.count("real"),
                  "rect_dropped": rect.get("removed_by_class", {}), "images": len(d["images"]),
                  "visibility_dropped": {k: v["dropped_objects"] for k, v in vis.items() if v.get("dropped_objects")},
                  "visibility_rejections": {k: v["rejected_attempts"] for k, v in vis.items() if v.get("rejected_attempts")}}
    return out


def stage_features(root, names):
    """Extract and cache the features (ResNet-50, CPU) of the variants. Resumable by chunks."""
    n_img = min(300, min(len(load_variant(Path(root) / n)[0]) for n in names))
    samp, quotas = av.sample_variants(root, names, n_img)
    ACACHE.mkdir(parents=True, exist_ok=True)
    model = dg.build_model()
    for n, s in samp.items():
        for kind in ("img", "obj"):
            np.save(ACACHE / f"{n}_{kind}.npy", dg.extract(f"variant_pilot/{n}_{kind}", s[kind], kind, model))
        json.dump(s, open(ACACHE / f"{n}_meta.json", "w", encoding="utf8"))
    json.dump({"n_img": n_img, "quotas": quotas}, open(ACACHE / "plan.json", "w"))


def note_lines(counts):
    """Report text about copy-paste and the discarded crops."""
    L = ["", "## What is measured in each variant", "",
         "- The object crops (AUC and diversity per object) include **only the pasted objects**; the kept real "
         "boxes of g, i and j (`origin = \"real\"`) are excluded.",
         "- g, i and j use real TRAIN backgrounds: their **image-level AUC is not comparable** with that of c, f and h "
         "(it drops trivially). The key metric is the **object-level AUC**.", "",
         "| Variant | Background | Images | Pasted objects | Kept real boxes | Rectangular crops discarded "
         "| Objects discarded by visibility (rejected attempts) |", "|---|---|---|---|---|---|---|"]
    for n, c in counts.items():
        rect = ", ".join(f"{k} {v}" for k, v in c["rect_dropped"].items()) or "-"
        vis = ", ".join(f"{k} {v} ({c['visibility_rejections'].get(k, 0)})"
                        for k, v in c["visibility_dropped"].items()) or "-"
        L.append(f"| {n} | {'real (TRAIN)' if n in REAL_BACKGROUND else 'synthetic'} | {c['images']} | "
                 f"{c['pasted']} | {c['kept_real']} | {rect} | {vis} |")
    return L


def write_md(rep, quotas, n_img, counts, path, meas=None):
    """Variant report with the notes about copy-paste and, if measured, sharpness/border/scale of the pasted objects."""
    av.write_md(rep, quotas, n_img, path, title="Pilot variants of the copy-paste batch (v11): small pilot, CPU")
    lines = Path(path).read_text(encoding="utf8").splitlines()
    extra = measures.md_lines(meas) if meas else []
    Path(path).write_text("\n".join(lines + note_lines(counts) + extra) + "\n", encoding="utf8")


def stage_analyze(root, names, baseline, reference=None):
    t0 = time.time()
    plan = json.load(open(ACACHE / "plan.json"))
    quotas, n_img = plan["quotas"], plan["n_img"]
    variants = av.build_variants(names, baseline, n_img, quotas)
    rep = av.analyze_variants(variants, av.real_subsets(n_img, quotas), quotas)
    counts = annotation_counts(root, names)
    meas = measures.measure_all(root, names, reference=reference)   # sharpness, noise, border and scale (CPU only)
    rep["_meta"] = {"n_img": n_img, "quotas": quotas, "seconds": round(time.time() - t0, 1), "counts": counts,
                    "measures": meas}
    (dg.WORK / "mosaics").mkdir(parents=True, exist_ok=True)
    for n in names:
        av.make_mosaic(n, dg.WORK / "mosaics" / f"variant_pilot_{n}.jpg")
    json.dump(rep, open(OUT / "variant_pilot.json", "w", encoding="utf8"), indent=1)
    write_md({k: v for k, v in rep.items() if k != "_meta"}, quotas, n_img, counts, OUT / "variant_pilot.md", meas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["features", "analyze", "all"], default="all")
    ap.add_argument("--root", default=str(av.PILOT_ROOT))
    ap.add_argument("--variants", nargs="*", default=None, help="default: those of VARIANTS whose folder exists in --root")
    ap.add_argument("--baseline", action="store_true", help="include the v9 batch of domain_gap (cache) as a reference")
    ap.add_argument("--train-reference", default=str(measures.DEFAULT_REFERENCE))
    a = ap.parse_args()
    if not a.variants:
        a.variants = [v for v in VARIANTS if (Path(a.root) / v / "synthetic_dataset.json").exists()]
    dg.lower_priority()
    if a.stage in ("features", "all"):
        stage_features(a.root, a.variants)
    if a.stage in ("analyze", "all"):
        stage_analyze(a.root, a.variants, a.baseline, a.train_reference)


if __name__ == "__main__":
    main()
