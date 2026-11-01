"""Domain gap between the clean-background synthetic batch (v9) and real images (TRAIN and VAL only, never TEST).

Measures whether a linear classifier separates synthetic from real images on the features of an ImageNet ResNet-50
from torchvision (global pooling, 224x224 squashed input, no center crop). This is not the Inception FID: the Frechet
distance and the KID are computed on the ResNet-50 features (2048 d).

Levels:
  * image: ~1000 synthetic vs ~1000 real TRAIN images (stratified by source), and per source.
  * object: box crops (+20 % context per side, 224 px) of ~1000 objects per side, stratified by class. Detects
    pasting artifacts.
  * control: real TRAIN vs real VAL with the same classifier (should give AUC ~0.5).
  * distances: Frechet and KID synthetic-real and real TRAIN-real VAL.
  * mosaics: the 24 synthetic images or crops that are most and least "detectable" (out-of-fold probability).

Usage (CPU only, idle priority, a single process):
  python analysis/synth_quality/domain_gap.py --stage features   # extract and cache .npy features
  python analysis/synth_quality/domain_gap.py --stage analyze    # metrics, json, md, mosaics
  python analysis/synth_quality/domain_gap.py --stage all
Reports go to analysis/synth_quality/out/; caches, samples and mosaics to RUNS_DIR/synth_quality/.
Environment: DATA_ROOT, RUNS_DIR (see common.paths) and SYNTH_BATCH (folder of the v9 batch with
synthetic_dataset.json and images/; default DATA_ROOT/synthetic/v9).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from common import paths  # noqa: E402

KIT = paths.KIT
OUT = HERE / "out"
WORK = paths.RUNS_DIR / "synth_quality"
CACHE = WORK / "cache"
DATA_ROOT = paths.DATA_ROOT
SYNTH_BATCH = Path(os.environ.get("SYNTH_BATCH", paths.SYNTH_DIR / "v9"))
CLASSES = paths.CLASSES
SEED = paths.SEED
CTX = 0.20      # context per side, fraction of the box side
MIN_BOX = 16    # minimum box side in pixels (in the original image)
SIZE = 224


# ------------------------------------------------------------------ sampling (pure, testable)
def allocate(counts, n, min_q=0):
    """Split n among strata: at least min(min_q, count) each and the rest proportional to what remains."""
    base = {k: min(min_q, c) for k, c in counts.items()}
    rem = min(n, sum(counts.values())) - sum(base.values())
    room = {k: counts[k] - base[k] for k in counts}
    tot = sum(room.values())
    if rem <= 0 or tot == 0:
        return base
    raw = {k: rem * room[k] / tot for k in room}
    out = {k: base[k] + int(raw[k]) for k in room}
    left = min(n, sum(counts.values())) - sum(out.values())
    for k in sorted(room, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if left <= 0:
            break
        if out[k] < counts[k]:
            out[k] += 1
            left -= 1
    return out


def stratified_sample(items, key, quotas, seed=SEED):
    """Take quotas[stratum] random elements from each stratum (deterministic)."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    out = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda x: json.dumps(x, sort_keys=True, default=str))
        rng.shuffle(g)
        out += g[:quotas.get(k, 0)]
    return out


def expand_box(bbox, w, h, ctx=CTX):
    """COCO box [x,y,w,h] enlarged by ctx*side on every side and clipped to the image -> (x0,y0,x1,y1)."""
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - ctx * bw), max(0, y - ctx * bh)
    x1, y1 = min(w, x + bw * (1 + ctx)), min(h, y + bh * (1 + ctx))
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


# ------------------------------------------------------------------ statistics (pure, testable)
def frechet_distance(a, b):
    """Frechet distance between Gaussians fitted to a and b (n x d); no Inception FID."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(s1)
    r1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = r1 @ s2 @ r1
    tr = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr)


def _poly(x, y):
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def kid(a, b, subsets=50, size=100, seed=SEED):
    """KID (unbiased MMD^2, cubic polynomial kernel) -> (mean, std) over `subsets` subsamples."""
    rng = np.random.default_rng(seed)
    size = min(size, len(a), len(b))
    vals = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), size, replace=False)]
        y = b[rng.choice(len(b), size, replace=False)]
        kxx, kyy, kxy = _poly(x, x), _poly(y, y), _poly(x, y)
        m = size
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def cv_classify(x, y, folds=5, seed=SEED, c=0.1, groups=None):
    """Standardized logistic regression with stratified CV (group-wise if `groups` is given).

    Returns per-fold metrics and the out-of-fold probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof, accs, aucs = np.zeros(len(y)), [], []
    if groups is None:
        splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y)
    else:
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(x, y, groups)
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])[:, 1]
        oof[te] = p
        accs.append(accuracy_score(y[te], p > 0.5))
        aucs.append(roc_auc_score(y[te], p))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "auc": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)), "n": int(len(y)), "oof": oof}


def subgroup_auc(oof, y, groups):
    """AUC of the out-of-fold probabilities inside each subgroup (class or source)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for g in sorted(set(groups)):
        m = np.array([gg == g for gg in groups])
        if len(set(y[m])) == 2:
            out[g] = {"auc": float(roc_auc_score(y[m], oof[m])), "n": int(m.sum()),
                      "acc": float(((oof[m] > 0.5) == (y[m] == 1)).mean())}
    return out


# ------------------------------------------------------------------ data loading
def _coco(path):
    return json.load(open(path, encoding="utf8"))


def load_real(split):
    """Real images and annotations of a split with their source and local path."""
    d = _coco(paths.annotations_path(split))
    names = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: {"id": im["id"], "path": str(DATA_ROOT / im["file_name"]), "source": im["source"],
                       "w": im["width"], "h": im["height"]} for im in d["images"]}
    anns = [{"img": a["image_id"], "bbox": a["bbox"], "cls": names[a["category_id"]]}
            for a in d["annotations"] if not a.get("iscrowd")]
    return imgs, anns


def load_synth():
    """Synthetic images and annotations of the v9 batch with their local path."""
    d = _coco(SYNTH_BATCH / "synthetic_dataset.json")
    names = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: {"id": im["id"], "path": str(SYNTH_BATCH / "images" / im["file_name"]), "source": "synthetic",
                       "w": im["width"], "h": im["height"]} for im in d["images"]}
    anns = [{"img": a["image_id"], "bbox": a["bbox"], "cls": names[a["category_id"]]}
            for a in d["annotations"] if not a.get("iscrowd", 0)]
    return imgs, anns


def sample_images(imgs, n, by_source, min_q=0, seed=SEED):
    """Sample n images (stratified by source if by_source) and check that the files exist."""
    items = [i for i in imgs.values() if os.path.exists(i["path"])]
    if not by_source:
        return stratified_sample(items, lambda i: "x", {"x": n}, seed)
    counts = {}
    for i in items:
        counts[i["source"]] = counts.get(i["source"], 0) + 1
    return stratified_sample(items, lambda i: i["source"], allocate(counts, n, min_q), seed)


def sample_objects(imgs, anns, quotas, seed=SEED):
    """Sample objects per class (valid boxes, existing image) with their image attached."""
    ok = [dict(a, **{"path": imgs[a["img"]]["path"], "w": imgs[a["img"]]["w"], "h": imgs[a["img"]]["h"]})
          for a in anns if a["bbox"][2] >= MIN_BOX and a["bbox"][3] >= MIN_BOX
          and os.path.exists(imgs[a["img"]]["path"])]
    return stratified_sample(ok, lambda a: a["cls"], quotas, seed), {c: sum(a["cls"] == c for a in ok) for c in CLASSES}


# ------------------------------------------------------------------ feature extraction
def load_pil(item, kind):
    """Open the image with JPEG draft mode and return the RGB PIL image (whole image or box crop)."""
    from PIL import Image
    im = Image.open(item["path"])
    if kind == "img":
        if im.format == "JPEG":
            im.draft("RGB", (SIZE, SIZE))
        return im.convert("RGB")
    x0, y0, x1, y1 = expand_box(item["bbox"], item["w"], item["h"])
    s = 1
    while s < 8 and min(x1 - x0, y1 - y0) / (s * 2) >= SIZE:
        s *= 2
    if im.format == "JPEG" and s > 1:
        im.draft("RGB", (im.size[0] // s, im.size[1] // s))
    f = im.size[0] / item["w"]
    return im.convert("RGB").crop((int(x0 * f), int(y0 * f), max(int(x1 * f), int(x0 * f) + 1),
                                   max(int(y1 * f), int(y0 * f) + 1)))


def build_model():
    """ImageNet ResNet-50 (torchvision weights), fc=Identity, 2 threads, CPU only."""
    import torch
    import torchvision
    torch.set_num_threads(2)
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval()


def _check_cache_items(d, items):
    """Invalidate cached chunks when the item list (same path+box+order) changed since they were extracted.

    Without this guard, changing the per-class quotas would reuse chunks of a different sample (rows misaligned with
    the metadata). A cache without fingerprint is accepted and the current fingerprint is written."""
    import hashlib
    h = hashlib.sha1(json.dumps([[i.get("path"), i.get("bbox")] for i in items]).encode()).hexdigest()
    f = d / "items.sha1"
    if f.exists() and f.read_text().strip() != h:
        for c in d.glob("c*.npy"):
            c.unlink()
    f.write_text(h)


def extract(name, items, kind, model, chunk=50):
    """Features (n x 2048) with chunked cache in CACHE/<name>/. Resumable."""
    import torch
    from torchvision import transforms as T
    d = CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    _check_cache_items(d, items)
    tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t0, parts = time.time(), []
    for k in range(0, len(items), chunk):
        f = d / f"c{k:05d}.npy"
        if not f.exists():
            x = torch.stack([tf(load_pil(it, kind)) for it in items[k:k + chunk]])
            with torch.no_grad():
                np.save(f, model(x).numpy().astype(np.float32))
        parts.append(np.load(f))
        print(f"[{name}] {min(k + chunk, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(parts)


# ------------------------------------------------------------------ stages
def make_samples():
    """Build (and save) the image and object samples. Returns a dict name -> (items, kind)."""
    sy_i, sy_a = load_synth()
    tr_i, tr_a = load_real("train")
    va_i, va_a = load_real("val")
    s = {"synth_img": (sample_images(sy_i, 1000, False), "img"),
         "real_train_img": (sample_images(tr_i, 1000, True, 150), "img"),
         "real_val_img": (sample_images(va_i, 1000, True, 100), "img")}
    _, c_sy = sample_objects(sy_i, sy_a, {c: 10 ** 9 for c in CLASSES})
    _, c_tr = sample_objects(tr_i, tr_a, {c: 10 ** 9 for c in CLASSES})
    _, c_va = sample_objects(va_i, va_a, {c: 10 ** 9 for c in CLASSES})
    q = {c: min(125, c_sy[c], c_tr[c]) for c in CLASSES}   # 125 per class (Metal_Debris: what is available)
    qv = {c: min(q[c], c_va[c]) for c in CLASSES}          # VAL has few Can/Metal_Debris
    s["synth_obj"] = (sample_objects(sy_i, sy_a, q)[0], "obj")
    s["real_train_obj"] = (sample_objects(tr_i, tr_a, q)[0], "obj")
    s["real_val_obj"] = (sample_objects(va_i, va_a, qv)[0], "obj")
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump({k: v[0] for k, v in s.items()}, open(WORK / "samples.json", "w", encoding="utf8"), indent=1)
    return s


def stage_features():
    """Extract and cache the features of the six samples."""
    s = make_samples()
    model = build_model()
    for name, (items, kind) in s.items():
        np.save(CACHE / f"{name}.npy", extract(name, items, kind, model))
        print(f"== {name}: {len(items)} items", flush=True)


def _xy(a, b):
    return np.concatenate([a, b]), np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)


def _m(r):
    return {k: v for k, v in r.items() if k != "oof"}


def controls(real_tr, real_va, g_tr, g_va):
    """Real-real controls: TRAIN vs VAL (plain and group-wise CV by partition group) and random halves of TRAIN."""
    xc, yc = _xy(real_tr, real_va)
    rng = np.random.default_rng(SEED)
    yr = rng.permutation(np.arange(len(real_tr)) % 2)
    return {"control_train_vs_val": _m(cv_classify(xc, yc)),
            "control_train_vs_val_grouped": _m(cv_classify(xc, yc, groups=np.array(g_tr + g_va))),
            "control_train_random_halves": _m(cv_classify(real_tr, yr))}


def level_report(synth, real_tr, real_va, meta_s, meta_tr, meta_va, group_key, gid):
    """Metrics of one level (image or object): global, per real subgroup, control and distances."""
    x, y = _xy(real_tr, synth)
    glob = cv_classify(x, y)
    groups_real = [group_key(m) for m in meta_tr]
    rep = {"global": {k: v for k, v in glob.items() if k != "oof"}}
    rep["per_real_group"] = group_auc_balanced(synth, real_tr, groups_real)
    gtr = [gid(m) for m in meta_tr]
    rep["global_grouped"] = _m(cv_classify(x, y, groups=np.array(gtr + [f"s{i}" for i in range(len(synth))])))
    rep.update(controls(real_tr, real_va, gtr, [gid(m) for m in meta_va]))
    rep["distances"] = {"frechet_synth_real": frechet_distance(synth, real_tr),
                        "frechet_real_train_val": frechet_distance(real_tr, real_va),
                        "kid_synth_real": kid(synth, real_tr), "kid_real_train_val": kid(real_tr, real_va)}
    return rep, glob["oof"]


def group_auc_balanced(synth, real, groups, seed=SEED):
    """Synthetic vs real inside each real group, with the synthetic set subsampled to the same size."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in sorted(set(groups)):
        idx = np.array([i for i, gg in enumerate(groups) if gg == g])
        sel = rng.choice(len(synth), min(len(idx), len(synth)), replace=False)
        x, y = _xy(real[idx], synth[sel])
        out[g] = {k: v for k, v in cv_classify(x, y).items() if k != "oof"}
    return out


def size_baseline(meta_s, meta_r):
    """Extra control: depth-3 tree on (w, h, aspect) to check whether the image size gives the synthetic set away."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier
    f = lambda m: [[i["w"], i["h"], i["w"] / i["h"]] for i in m]
    x, y = _xy(np.array(f(meta_r), float), np.array(f(meta_s), float))
    p = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=SEED), x, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, p)), "acc": float(((p > 0.5) == y).mean())}


# ------------------------------------------------------------------ mosaics
def _tile(item, kind, cap, w=300, h=225):
    from PIL import Image, ImageDraw
    im = load_pil(item, kind)
    im.thumbnail((w, h))
    t = Image.new("RGB", (w, h + 14), (30, 30, 30))
    t.paste(im, ((w - im.size[0]) // 2, 0))
    ImageDraw.Draw(t).text((3, h + 1), cap, fill=(255, 255, 0))
    return t


def mosaic(items, kind, probs, order, path, cols=6):
    """JPEG mosaic (<= 2000 px) with the items in `order`; the caption carries the out-of-fold P(synthetic)."""
    from PIL import Image
    tiles = [_tile(items[i], kind, f"P(synth)={probs[i]:.2f}" + (f" {items[i]['cls']}" if kind == "obj" else ""))
             for i in order]
    rows = (len(tiles) + cols - 1) // cols
    m = Image.new("RGB", (cols * 300, rows * 239))
    for k, t in enumerate(tiles):
        m.paste(t, ((k % cols) * 300, (k // cols) * 239))
    m.save(path, quality=80)


def save_mosaics(tag, items, kind, oof_synth):
    """Save the mosaics of the 24 most and least detectable items (oof_synth: P(synthetic) of the synthetic set)."""
    d = WORK / "mosaics"
    d.mkdir(parents=True, exist_ok=True)
    o = np.argsort(oof_synth)
    mosaic(items, kind, oof_synth, o[::-1][:24], d / f"mosaic_{tag}_most_detectable.jpg")
    mosaic(items, kind, oof_synth, o[:24], d / f"mosaic_{tag}_least_detectable.jpg")


# ------------------------------------------------------------------ report
def fmt(m):
    return f"{m['acc']:.3f} ± {m['acc_std']:.3f} / AUC {m['auc']:.3f} ± {m['auc_std']:.3f} (n={m['n']})"


def write_md(r, path):
    L = ["# Domain gap: v9 synthetic batch vs real", "",
         "Features: ImageNet ResNet-50 (torchvision, global pooling, 224x224). Classifier: standardized logistic "
         "regression (C=0.1), 5-fold stratified CV. Frechet and KID on those features (not the Inception FID). "
         "TRAIN/VAL only.", ""]
    for lvl in ("image", "object"):
        d = r[lvl]
        L += [f"## {lvl.capitalize()} level", f"- Synthetic vs real TRAIN: {fmt(d['global'])}",
              f"- Synthetic vs real TRAIN, CV grouped by partition group: {fmt(d['global_grouped'])}",
              f"- **Control** real TRAIN vs real VAL: {fmt(d['control_train_vs_val'])}",
              f"- Control TRAIN vs VAL with grouped CV (no scene overlap): {fmt(d['control_train_vs_val_grouped'])}",
              f"- Control TRAIN split at random into two halves: {fmt(d['control_train_random_halves'])}",
              "- Per real group:"]
        L += [f"  - {g}: {fmt(m)}" for g, m in d["per_real_group"].items()]
        if "per_class_oof" in d:
            L += ["- Per class (out-of-fold probabilities of the global classifier):"]
            L += [f"  - {g}: AUC {m['auc']:.3f}, acc {m['acc']:.3f} (n={m['n']})" for g, m in d["per_class_oof"].items()]
        dd = d["distances"]
        L += [f"- Frechet: synth-real {dd['frechet_synth_real']:.1f}; real TRAIN-VAL {dd['frechet_real_train_val']:.1f}",
              f"- KID: synth-real {dd['kid_synth_real'][0]:.4f} ± {dd['kid_synth_real'][1]:.4f}; "
              f"real TRAIN-VAL {dd['kid_real_train_val'][0]:.4f} ± {dd['kid_real_train_val'][1]:.4f}", ""]
    L += [f"## Image size control (depth-3 tree on w, h, aspect): AUC {r['image_size']['auc']:.3f}, "
          f"acc {r['image_size']['acc']:.3f}", "", f"Analysis time: {r['analysis_time_s']:.0f} s"]
    open(path, "w", encoding="utf8").write("\n".join(L) + "\n")


def group_ids():
    """Map real image id -> group of the partition (near-duplicates and scenes stay together)."""
    g = json.load(open(paths.SPLIT_OUT / "groups.json", encoding="utf8"))
    return {i: f"g{x['group']}" for x in g for i in x["image_ids"]}


def stage_analyze():
    """Compute the metrics from the cache and save json/md and mosaics."""
    t0 = time.time()
    s = json.load(open(WORK / "samples.json", encoding="utf8"))
    f = {k: np.load(CACHE / f"{k}.npy") for k in s}
    gm = group_ids()
    r = {}
    for lvl, suf, key in (("image", "img", lambda m: m["source"]), ("object", "obj", lambda m: m["cls"])):
        rep, oof = level_report(f[f"synth_{suf}"], f[f"real_train_{suf}"], f[f"real_val_{suf}"],
                                s[f"synth_{suf}"], s[f"real_train_{suf}"], s[f"real_val_{suf}"], key,
                                lambda m: gm[m["id"] if suf == "img" else m["img"]])
        n_r = len(f[f"real_train_{suf}"])
        meta_all = s[f"real_train_{suf}"] + s[f"synth_{suf}"]
        y = np.r_[np.zeros(n_r), np.ones(len(f[f"synth_{suf}"]))].astype(int)
        if suf == "obj":
            rep["per_class_oof"] = subgroup_auc(oof, y, [m["cls"] for m in meta_all])
        r[lvl] = rep
        save_mosaics(suf, s[f"synth_{suf}"], suf, oof[n_r:])
    r["image_size"] = size_baseline(s["synth_img"], s["real_train_img"])
    r["analysis_time_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(OUT / "domain_gap.json", "w", encoding="utf8"), indent=1)
    write_md(r, OUT / "domain_gap.md")


def lower_priority():
    """Lowest process priority so that other jobs on the machine are not disturbed."""
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception as e:  # noqa: BLE001
        print("warning: could not lower the process priority:", e, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["features", "analyze", "all"], default="all")
    a = ap.parse_args()
    lower_priority()
    if a.stage in ("features", "all"):
        stage_features()
    if a.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
