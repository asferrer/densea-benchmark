"""Library of the pilot-variant metrics: realism and diversity of each variant of a small pilot (CPU only).

Per variant (folders `<PILOT_ROOT>/<variant>/` with images/ and synthetic_dataset.json written by the generator):
  * realism: synthetic-vs-real AUC (ResNet-50 + logistic regression, same recipe as domain_gap.py) per image and per
    object, and the Frechet distance on the same features. Averaged over 3 CV seeds.
  * object diversity, per class: Vendi score (q=1, cosine similarity) and mean cosine distance to the nearest neighbor
    among the object-crop features. Compared with the real TRAIN objects with the SAME number of items per class (the
    Vendi score grows with n).
GROUPED AUC (main column): CV with StratifiedGroupKFold. Groups: real images (TRAIN and VAL) = partition group of the
image (all the objects of an image share a group); synthetic images with a real TRAIN background (copy-paste) =
partition group of the real background image (COCO field `background` -> TRAIN annotations -> id -> group), so that
synthetic and real items of the same group are never split across folds; synthetic images without a real background =
one group per reused background (one per synthetic image if the generator writes no `background`). If a real-background
variant has no traceable background, one group per image is used and the report says so. The ungrouped AUC is kept as a
reference. Real items always come from the domain_gap cache (RUNS_DIR/synth_quality/cache/real_*.npy), with the same
subset for every variant (same size as the synthetic one). TRAIN and VAL only, never TEST. Control: TRAIN vs VAL with
the same sizes.

Copy-paste variants keep the real boxes of the background image in their COCO (`origin = "real"`); the object crops of
all variants contain only the PASTED objects (`origin = "pasted"` or no `origin`).
Batch labels: v9 = clean-background batch, v11 = copy-paste batch.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import domain_gap as dg  # noqa: E402
from common import paths  # noqa: E402

OUT, ACACHE = dg.OUT, dg.CACHE / "variant_pilot"
PILOT_ROOT = Path(os.environ.get("PILOT_ROOT", paths.RUNS_DIR / "pilots"))
BASELINE = "ref_batch_v9"
SEEDS = (dg.SEED, dg.SEED + 1, dg.SEED + 2)
CAP = 125   # maximum objects per class (same as domain_gap)


# ------------------------------------------------------------------ pure metrics (testable)
def _unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def vendi_score(x):
    """Vendi score (q=1) with cosine similarity: effective number of distinct items (1..n)."""
    z = _unit(np.asarray(x, dtype=np.float64))
    w = np.clip(np.linalg.eigvalsh(z @ z.T / len(z)), 0, None)
    w = w[w > 1e-12] / w[w > 1e-12].sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def mean_nn_distance(x):
    """Mean cosine distance (1 - cos) of each item to its nearest neighbor in the same set."""
    z = _unit(np.asarray(x, dtype=np.float64))
    s = z @ z.T
    np.fill_diagonal(s, -np.inf)
    return float((1.0 - s.max(axis=1)).mean())


def common_quota(counts_list, cap=CAP):
    """Per-class quota: the minimum over all sets (same n per class in every variant)."""
    return {c: min([cap] + [cnt.get(c, 0) for cnt in counts_list]) for c in dg.CLASSES}


def pick_indices(meta, key, n, seed=dg.SEED):
    """Sorted, deterministic indices of n items stratified by key(meta)."""
    counts = {}
    for m in meta:
        counts[key(m)] = counts.get(key(m), 0) + 1
    quota = dg.allocate(counts, n)
    rng, out = np.random.default_rng(seed), []
    for k in sorted(counts):
        idx = [i for i, m in enumerate(meta) if key(m) == k]
        out += list(rng.permutation(idx)[:quota.get(k, 0)])
    return np.array(sorted(int(i) for i in out), dtype=int)


def pick_by_class(meta, quotas, seed=dg.SEED):
    """Sorted indices with quotas[class] random items (deterministic) of each class."""
    rng, out = np.random.default_rng(seed), []
    for c in dg.CLASSES:
        idx = [i for i, m in enumerate(meta) if m["cls"] == c]
        out += list(rng.permutation(idx)[:quotas.get(c, 0)])
    return np.array(sorted(int(i) for i in out), dtype=int)


# ------------------------------------------------------------------ groups of the grouped CV (pure, testable)
REAL_BACKGROUND = {"g_v11_copypaste", "i_v11_all", "j_v11_visible", "k_sharp", "l_scale", "m_pos", "n_vis15",
                   "o_tight", "p_pos_tight", "q_base", "q_lab", "q_lab_same", "q_lab_local", "r_matte", "r_matte_fg",
                   "r_medium", "r_medium_harm", "s_clean", "s_drop"}


class _RealBg(set):
    """Names of the variants whose background is a real TRAIN image. Folders of the diffusion-refined variants
    (`p_pos_tight_r030_c10`, `p_pos_tight_ctrl`, ...) start from p_pos_tight and also count."""
    PREFIXES = ("p_pos_tight_",)

    def __contains__(self, name):
        return set.__contains__(self, name) or str(name).startswith(self.PREFIXES)


REAL_BG_VARIANTS = _RealBg(REAL_BACKGROUND)


def background_group_map(annotations_file=None, gmap=None):
    """File name of a real TRAIN background image -> partition group of that image.

    The generator writes in `background` the file name of the real TRAIN image; both the full `file_name` of the TRAIN
    annotations and its base name are accepted."""
    ims = json.load(open(annotations_file or paths.annotations_path("train"), encoding="utf8"))["images"]
    gmap = dg.group_ids() if gmap is None else gmap
    out = {}
    for i in ims:
        if i["id"] in gmap:
            out[i["file_name"]] = gmap[i["id"]]
            out.setdefault(Path(i["file_name"]).name, gmap[i["id"]])
    return out


def synth_image_groups(variant, images, bg_map, real_bg):
    """Group of each synthetic image: {id: group}, and info {criterion, traceable, n_images, n_groups}.

    images: [{"id", "background"?}]. Traceable real background -> partition group of the real image; no real background
    -> one group per reused background (or per image without identifier); real-background variant not traceable -> per
    image (annotated)."""
    out, traced = {}, 0
    for im in images:
        bg = im.get("background")
        if bg in bg_map:
            out[im["id"]] = bg_map[bg]
            traced += 1
        elif real_bg or not bg:
            out[im["id"]] = f"s:{variant}:img{im['id']}"
        else:
            out[im["id"]] = f"s:bg:{bg}"
    n = len(images)
    if traced:
        crit = "partition group of the real background image" + ("" if traced == n else f" ({n - traced} not traceable: per image)")
    elif real_bg:
        crit = "NOT TRACEABLE: one group per synthetic image"
    elif any(im.get("background") for im in images):
        crit = "one group per background (not a real TRAIN image)"
    else:
        crit = "one group per synthetic image (no background identifier)"
    return out, {"criterion": crit, "traceable": traced, "n_images": n, "n_groups": len(set(out.values()))}


def variant_images(item_paths, variant):
    """Images (id, background) of a variant read from the COCO next to the sample paths.

    If the COCO does not exist an empty dict is returned (every group then falls back to 'per image')."""
    dirs = {Path(p).parents[1] for p in item_paths}
    out = {}
    for d in dirs:
        f = d / "synthetic_dataset.json"
        if f.exists():
            for im in json.load(open(f, encoding="utf8"))["images"]:
                out[im["id"]] = {"id": im["id"], "background": im.get("background")}
    return out


def attach_groups(v, name, bg_map):
    """Add v['img_g'] and v['obj_g'] (group per row) and v['groups_info'] to a variant."""
    img_meta, obj_meta = v["img_meta"], v["obj_meta"]
    ids = {m["id"] for m in img_meta} | {m["img"] for m in obj_meta}
    known = variant_images([m["path"] for m in img_meta + obj_meta], name)
    images = [known.get(i, {"id": i}) for i in sorted(ids)]
    gm, info = synth_image_groups(name, images, bg_map, name in REAL_BG_VARIANTS)
    v["img_g"] = [gm[m["id"]] for m in img_meta]
    v["obj_g"] = [gm[m["img"]] for m in obj_meta]
    v["groups_info"] = info
    return v


def real_groups(meta, key):
    """Partition group of each real row (key = 'id' for images, 'img' for objects)."""
    gm = dg.group_ids()
    return [gm[m[key]] for m in meta]


def auc_report(synth, real, g_synth=None, g_real=None):
    """Mean AUC/acc (3 CV seeds) of real (0) vs synthetic (1) and the Frechet distance between them.

    Ungrouped: `auc`, `auc_std`, `acc`. With groups (g_synth, g_real) also `auc_agr`, `auc_agr_std`, `acc_agr`
    (StratifiedGroupKFold)."""
    x, y = dg._xy(real, synth)
    runs = [dg.cv_classify(x, y, seed=s) for s in SEEDS]
    rep = {"auc": float(np.mean([r["auc"] for r in runs])), "auc_std": float(np.std([r["auc"] for r in runs])),
           "acc": float(np.mean([r["acc"] for r in runs])), "n_synth": int(len(synth)), "n_real": int(len(real)),
           "frechet": dg.frechet_distance(synth, real)}
    if g_synth is not None and g_real is not None:
        grp = np.array(list(g_real) + list(g_synth))
        gr = [dg.cv_classify(x, y, seed=s, groups=grp) for s in SEEDS]
        rep.update(auc_agr=float(np.mean([r["auc"] for r in gr])), auc_agr_std=float(np.std([r["auc"] for r in gr])),
                   acc_agr=float(np.mean([r["acc"] for r in gr])), n_groups=int(len(set(grp))))
    return rep


def diversity_by_class(feats, meta, quotas):
    """Vendi and nearest-neighbor distance per class (n = quota) and their mean over classes."""
    out = {}
    for c in dg.CLASSES:
        idx = [i for i, m in enumerate(meta) if m["cls"] == c][:quotas[c]]
        if len(idx) >= 3:
            out[c] = {"n": len(idx), "vendi": vendi_score(feats[idx]), "nn_dist": mean_nn_distance(feats[idx])}
    if out:
        out["mean"] = {k: float(np.mean([v[k] for v in out.values()])) for k in ("vendi", "nn_dist")}
    return out


# ------------------------------------------------------------------ loading variants
def keep_pasted(ann):
    """True if the annotation is a pasted object (the kept real boxes carry origin = 'real')."""
    return ann.get("origin", "pasted") != "real"


def load_variant(vdir):
    """Images and PASTED annotations of a variant (generator COCO); excludes the kept real boxes."""
    d = json.load(open(Path(vdir) / "synthetic_dataset.json", encoding="utf8"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: {"id": im["id"], "path": str(Path(vdir) / "images" / im["file_name"]), "source": "synthetic",
                       "w": im["width"], "h": im["height"]} for im in d["images"]}
    anns = [{"img": a["image_id"], "bbox": a["bbox"], "cls": names[a["category_id"]]}
            for a in d["annotations"] if not a.get("iscrowd", 0) and keep_pasted(a)]
    return imgs, anns


def real_reference():
    """Real samples and features cached by domain_gap (TRAIN and VAL)."""
    s = json.load(open(dg.WORK / "samples.json", encoding="utf8"))
    f = {k: np.load(dg.CACHE / f"{k}.npy") for k in s if k.startswith("real")}
    return s, f


def sample_variants(root, names, n_img):
    """Per variant: images (first n_img by path) and objects with a common per-class quota."""
    data = {n: load_variant(Path(root) / n) for n in names}
    counts = {n: dg.sample_objects(i, a, {c: 10 ** 9 for c in dg.CLASSES})[1] for n, (i, a) in data.items()}
    s, _ = real_reference()
    real_counts = {c: sum(m["cls"] == c for m in s["real_train_obj"]) for c in dg.CLASSES}
    quotas = common_quota(list(counts.values()) + [real_counts])
    out = {}
    for n, (imgs, anns) in data.items():
        items = sorted((i for i in imgs.values() if os.path.exists(i["path"])), key=lambda i: i["path"])[:n_img]
        out[n] = {"img": items, "obj": dg.sample_objects(imgs, anns, quotas)[0]}
    return out, quotas


def baseline_data(n_img, quotas):
    """The v9 batch of domain_gap as one more variant (cache only; no image is opened)."""
    sy = json.load(open(dg.WORK / "samples.json", encoding="utf8"))
    fi, fo = np.load(dg.CACHE / "synth_img.npy"), np.load(dg.CACHE / "synth_obj.npy")
    ii = pick_indices(sy["synth_img"], lambda m: "x", n_img)
    io = pick_by_class(sy["synth_obj"], quotas)
    return {"img": fi[ii], "obj": fo[io], "obj_meta": [sy["synth_obj"][i] for i in io],
            "img_meta": [sy["synth_img"][i] for i in ii]}


def variant_data(name, cache=None):
    """Cached features and metadata of a variant."""
    cache = Path(cache or ACACHE)
    meta = json.load(open(cache / f"{name}_meta.json", encoding="utf8"))
    return {"img": np.load(cache / f"{name}_img.npy"), "obj": np.load(cache / f"{name}_obj.npy"),
            "obj_meta": meta["obj"], "img_meta": meta["img"]}


def real_subsets(n_img, quotas):
    """Same real subset for every variant: TRAIN and VAL (image) and TRAIN per class (object)."""
    s, f = real_reference()
    key = lambda m: m["source"]  # noqa: E731
    tr_i, va_i = (pick_indices(s[k], key, n_img) for k in ("real_train_img", "real_val_img"))
    tr_o = pick_by_class(s["real_train_obj"], quotas)
    qv = {c: min(quotas[c], sum(m["cls"] == c for m in s["real_val_obj"])) for c in dg.CLASSES}
    va_o = pick_by_class(s["real_val_obj"], qv)
    mt, mv = [s["real_train_img"][i] for i in tr_i], [s["real_val_img"][i] for i in va_i]
    ot, ov = [s["real_train_obj"][i] for i in tr_o], [s["real_val_obj"][i] for i in va_o]
    return {"img": f["real_train_img"][tr_i], "img_val": f["real_val_img"][va_i], "obj": f["real_train_obj"][tr_o],
            "obj_val": f["real_val_obj"][va_o], "obj_meta": ot,
            "img_g": real_groups(mt, "id"), "img_val_g": real_groups(mv, "id"),
            "obj_g": real_groups(ot, "img"), "obj_val_g": real_groups(ov, "img")}


def build_variants(names, baseline, n_img, quotas):
    """Data (features, metadata and CV groups) of the variants and, if requested, of the v9 reference batch."""
    bg_map = background_group_map()
    variants = {n: variant_data(n) for n in names}
    if baseline:
        variants = {BASELINE: baseline_data(n_img, quotas), **variants}
    return {n: attach_groups(v, n, bg_map) for n, v in variants.items()}


# ------------------------------------------------------------------ analysis and report
def analyze_variants(variants, real, quotas):
    """Report per variant plus the real reference (control and real diversity)."""
    rep = {"real_ref": {
        "control_image": auc_report(real["img_val"], real["img"], real.get("img_val_g"), real.get("img_g")),
        "control_object": auc_report(real["obj_val"], real["obj"], real.get("obj_val_g"), real.get("obj_g")),
        "object_diversity": diversity_by_class(real["obj"], real["obj_meta"], quotas)}}
    for n, v in variants.items():
        rep[n] = {"image": auc_report(v["img"], real["img"], v.get("img_g"), real.get("img_g")),
                  "object": auc_report(v["obj"], real["obj"], v.get("obj_g"), real.get("obj_g")),
                  "groups": v.get("groups_info"), "object_diversity": diversity_by_class(v["obj"], v["obj_meta"], quotas)}
    return rep


def group_lines(r):
    """Report section with the group criterion of each variant (and whether the background was not traceable)."""
    L = ["## Groups of the grouped CV", "",
         "- Real images (TRAIN and VAL): partition group of the image; the objects of an image share a group.",
         "- Synthetic with a real TRAIN background: partition group of the real background image (COCO field "
         "`background`), so synthetic and real items of the same group are never split across folds.",
         "- Synthetic without a real background: one group per reused background or, without identifier, one per "
         "synthetic image.", "",
         "| Variant | Criterion | Traceable backgrounds | Images | Groups (with real) |", "|---|---|---|---|---|"]
    for n, v in r.items():
        g = v.get("groups") if n != "real_ref" else None
        if g:
            L.append(f"| {n} | {g['criterion']} | {g['traceable']} | {g['n_images']} | {v['object'].get('n_groups', 'n/a')} |")
    return L + [""]


def _auc_cells(d):
    """'grouped ± std | ungrouped ± std' of an auc_report (grouped = n/a if it was not computed)."""
    g = f"{d['auc_agr']:.3f} ± {d['auc_agr_std']:.3f}" if "auc_agr" in d else "n/a"
    return f"{g} | {d['auc']:.3f} ± {d['auc_std']:.3f}"


def write_md(r, quotas, n_img, path, title="Pilot variants (small pilot, CPU)"):
    L = [f"# {title}", "",
         f"Images per variant: {n_img}. Objects per class (common quota): {quotas}. Real: same TRAIN subset for all "
         "variants. AUC: mean ± standard deviation over 3 CV seeds (5 folds). **Main column: AUC with GROUPED CV** "
         "(StratifiedGroupKFold; groups in the \"Groups\" section); the AUC with stratified CV without grouping is kept "
         "next to it as a reference. Frechet on ResNet-50 (2048 d) with small n: biased upwards, only comparable "
         "between variants of this same report.", "",
         "| Variant | AUC img grouped | AUC img ungrouped | Frechet img | AUC obj grouped | AUC obj ungrouped "
         "| Frechet obj | Mean Vendi | Mean NN |", "|---|---|---|---|---|---|---|---|---|"]
    for n, v in r.items():
        if n == "real_ref":
            continue
        d = v["object_diversity"].get("mean", {})
        L.append(f"| {n} | {_auc_cells(v['image'])} | {v['image']['frechet']:.1f} | "
                 f"{_auc_cells(v['object'])} | {v['object']['frechet']:.1f} | "
                 f"{d.get('vendi', float('nan')):.2f} | {d.get('nn_dist', float('nan')):.4f} |")
    ref = r["real_ref"]
    rd = ref["object_diversity"].get("mean", {})
    ci, co = ref["control_image"], ref["control_object"]
    L += ["", "Control real TRAIN vs VAL (grouped | ungrouped): "
          f"img {_auc_cells(ci)}; obj {_auc_cells(co)}. "
          f"Frechet real TRAIN-VAL: img {ci['frechet']:.1f}, obj {co['frechet']:.1f}.",
          f"Diversity of real TRAIN objects (same n per class): "
          f"mean Vendi {rd.get('vendi', float('nan')):.2f}, mean NN {rd.get('nn_dist', float('nan')):.4f}.", ""]
    L += group_lines(r)
    L += _class_diversity_lines(r)
    open(path, "w", encoding="utf8").write("\n".join(L) + "\n")


def _class_diversity_lines(r):
    ref = r["real_ref"]
    L = ["## Diversity per class (Vendi / NN)", "", "| Class | n | real | " + " | ".join(
        k for k in r if k != "real_ref") + " |", "|---|---|---|" + "---|" * (len(r) - 1)]
    for c in dg.CLASSES:
        if c in ref["object_diversity"]:
            cells = [f"{ref['object_diversity'][c]['vendi']:.1f} / {ref['object_diversity'][c]['nn_dist']:.3f}"]
            for k, v in r.items():
                if k != "real_ref":
                    d = v["object_diversity"].get(c)
                    cells.append(f"{d['vendi']:.1f} / {d['nn_dist']:.3f}" if d else "-")
            L.append(f"| {c} | {ref['object_diversity'][c]['n']} | " + " | ".join(cells) + " |")
    return L


def make_mosaic(name, path, n_imgs=8, n_objs=4, cache=None):
    """Small mosaic: the first images of the variant (4 per row) and a few object crops."""
    from PIL import Image
    meta = json.load(open(Path(cache or ACACHE) / f"{name}_meta.json", encoding="utf8"))
    tiles = [dg._tile(i, "img", "", 240, 180) for i in meta["img"][:n_imgs]]
    tiles += [dg._tile(o, "obj", o["cls"], 240, 180) for o in meta["obj"][:: max(1, len(meta["obj"]) // n_objs)][:n_objs]]
    cols = 4
    m = Image.new("RGB", (cols * 240, ((len(tiles) + cols - 1) // cols) * 194))
    for k, t in enumerate(tiles):
        m.paste(t, ((k % cols) * 240, (k // cols) * 194))
    m.save(path, quality=80)
