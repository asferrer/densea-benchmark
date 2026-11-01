"""Strategies that combine a real training set with a synthetic batch.

  stratified_subset     class-stratified sub-sampling of the real set (deterministic for a seed);
  minority_focus (MF)   synthetic images that contain the minority classes (Can, Metal_Debris) until each
                        class reaches MF_TARGET boxes together with the real ones;
  inverse_proportional (IP)
                        for every class, from the largest deficit to the smallest, synthetic images until
                        the class reaches IP_TARGET boxes with the real ones; no image is used twice;
  merge                 synthetic images followed by the real ones (ids shifted), with a `source` field.

Rules shared by MF and IP:
  - the deficit of a class is computed from the box counts of the real set they receive;
  - synthetic images without annotations are not used (dropped when the batch is loaded);
  - copy-paste batches ("v11") keep the real objects of the background image (`origin` = "real"): only
    pasted objects count towards the deficit and make an image a candidate, and the real boxes stay in
    the image. Batches without an `origin` field (clean-background batch, "v9") count every object as pasted.
Batch labels: v9 is the clean-background batch, v11 the copy-paste batch.
Each strategy draws from its own ``random.Random(seed)``, so there is no global random state.
"""
import collections
import random
from copy import deepcopy

MINORITY_CLASSES = ["Can", "Metal_Debris"]
MF_TARGET = 3500
IP_TARGET = 4500


def by_category(ds, pasted_only=False):
    """category_id -> set of image ids that contain it (with pasted_only, only as a pasted object)."""
    out = collections.defaultdict(set)
    for a in ds["annotations"]:
        if not pasted_only or is_pasted(a):
            out[a["category_id"]].add(a["image_id"])
    return out


def class_counts(ds):
    names = {c["id"]: c["name"] for c in ds["categories"]}
    return collections.Counter(names[a["category_id"]] for a in ds["annotations"])


def filter_images(ds, keep):
    keep = set(keep)
    return {**{k: v for k, v in ds.items() if k not in ("images", "annotations")},
            "images": [im for im in ds["images"] if im["id"] in keep],
            "annotations": [a for a in ds["annotations"] if a["image_id"] in keep]}


def stratified_subset(ds, ratio, seed):
    """Class-stratified sub-sampling of ``ds`` to a fraction ``ratio`` of its images."""
    if ratio >= 1.0:
        return deepcopy(ds)
    rng = random.Random(seed)
    cat_imgs = by_category(ds)
    all_ids = {im["id"] for im in ds["images"]}
    target = int(len(all_ids) * ratio)
    sel = set()
    for _cat, imgs in cat_imgs.items():
        lst = list(imgs)
        avail = [i for i in lst if i not in sel]
        if avail:
            sel.update(rng.sample(avail, min(max(1, int(len(lst) * ratio)), len(avail))))
    if len(sel) < target:
        rest = list(all_ids - sel)
        sel.update(rng.sample(rest, min(target - len(sel), len(rest))))
    if len(sel) > target:
        sel = set(rng.sample(list(sel), target))
    return filter_images(ds, sel)


def is_pasted(a):
    """True for an object pasted by the generator (background objects, origin = "real", are not)."""
    return a.get("origin", "pasted") != "real"


def _pick_until(rng, candidates, img_anns, cat_id, needed, already=()):
    """Add shuffled images until `needed` PASTED objects of `cat_id` are accumulated."""
    cand = [i for i in candidates if i not in already]
    rng.shuffle(cand)
    chosen, added = [], 0
    for i in cand:
        if added >= needed:
            break
        chosen.append(i)
        added += sum(1 for a in img_anns[i] if a["category_id"] == cat_id and is_pasted(a))
    return chosen, added


def _img_anns(ds):
    out = collections.defaultdict(list)
    for a in ds["annotations"]:
        out[a["image_id"]].append(a)
    return out


def minority_focus(real, synth, seed, target=MF_TARGET, classes=MINORITY_CLASSES):
    rng = random.Random(seed)
    ids = {c["name"]: c["id"] for c in synth["categories"]}
    real_n, cat_imgs, img_anns = class_counts(real), by_category(synth, pasted_only=True), _img_anns(synth)
    sel, log = set(), {}
    for name in classes:
        needed = max(0, target - real_n.get(name, 0))
        chosen, added = _pick_until(rng, list(cat_imgs.get(ids[name], ())), img_anns, ids[name], needed)
        sel.update(chosen)
        log[name] = {"real": real_n.get(name, 0), "needed": needed, "added": added}
    return filter_images(synth, sel), log


def inverse_proportional(real, synth, seed, target=IP_TARGET):
    rng = random.Random(seed)
    ids = {c["name"]: c["id"] for c in synth["categories"]}
    real_n, cat_imgs, img_anns = class_counts(real), by_category(synth, pasted_only=True), _img_anns(synth)
    deficit = {n: max(0, target - real_n.get(n, 0)) for n in ids}
    sel, log = set(), {}
    for name, needed in sorted(deficit.items(), key=lambda x: -x[1]):
        if needed <= 0:
            continue
        chosen, added = _pick_until(rng, list(cat_imgs.get(ids[name], ())), img_anns, ids[name], needed, sel)
        sel.update(chosen)
        log[name] = {"real": real_n.get(name, 0), "needed": needed, "added": added}
    return filter_images(synth, sel), log


def merge(synth, real, name, lot="v9"):
    """Synthetic images first (ids unchanged) + real images with shifted ids; `source` in every image."""
    synth, real = deepcopy(synth), deepcopy(real)
    for im in synth["images"]:
        im["source"] = "synthetic"
    for im in real["images"]:
        im["source"] = "real"
    off_i = max((im["id"] for im in synth["images"]), default=-1) + 1
    off_a = max((a["id"] for a in synth["annotations"]), default=-1) + 1
    for im in real["images"]:
        im["id"] += off_i
    for a in real["annotations"]:
        a["id"] += off_a
        a["image_id"] += off_i
    return {"info": {"description": f"{name} (group partition + synthetic batch {lot})"}, "licenses": [],
            "categories": real["categories"], "images": synth["images"] + real["images"],
            "annotations": synth["annotations"] + real["annotations"]}
