"""Registry of feature extractors for the gap panel (CPU only) and their disk cache.

Adding an extractor = adding ONE entry to REGISTRY (id, license, normalization and `build` function).
Each crop is opened once (`domain_gap.load_pil`, box enlarged by CTX 0.20), squashed to 224x224 (like the ResNet-50 of
domain_gap) and passed through all the requested extractors. Resumable chunk cache in
`RUNS_DIR/synth_quality/cache/gap_panel/<extractor>/<set>/`, assembled in `.../gap_panel/<extractor>/<set>.npy`.
The ResNet-50 reuses, when the rows match, the caches that already exist (`shared_cache`).

If a model cannot be loaded (e.g. a gated repository without access) `available()` reports it and the panel goes on
without it.

Task evaluator (`rtdetrv2_real10`): RT-DETRv2 R50 trained on 10 % of the real TRAIN images (`px_real10`), run on CPU
from the RT-DETRv2 repository. Main layer = output P3 of the HybridEncoder (stride 8, 256 d), mean inside the object
box: it is the highest-resolution map consumed by the decoder, it already fuses the backbone with context through
the top-down and bottom-up branches, and it is the last thing the detector sees before deciding class and box. With
the crop (box +20 % per side) squashed to 640x640 the object covers ~71 % of the side: ~56x56 cells of P3 (~14x14 of P5)
contribute to the mean.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import contextlib
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import domain_gap as dg  # noqa: E402
from common import paths  # noqa: E402

CACHE = CACHE_MAIN = dg.CACHE / "gap_panel"
SIZE = 224
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_MS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


@dataclass
class Spec:
    name: str
    hf_id: str
    license: str
    mean_std: tuple
    build: Callable            # () -> function (B,3,224,224) -> (B,d)
    notes: str = ""
    gated: bool = False
    shared_cache: bool = False  # True: reuses the domain_gap/variant_pilot caches when the rows match
    dim: int = 0
    size: int = SIZE           # side of the squashed crop the extractor receives (RT-DETRv2: 640, as in VAL)
    uses_rois: bool = False    # True: fn(x, rois) with the object box inside the crop, normalized (B,4) in [0,1]
    _fn: object = field(default=None, repr=False)

    def fn(self):
        if self._fn is None:
            self._fn = self.build()
        return self._fn


def _threads():
    import torch
    torch.set_num_threads(int(os.environ.get("GAP_THREADS", "8")))
    return torch


def _build_resnet50():
    import torchvision
    torch = _threads()
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    return lambda x: m(x)


def _hf_path(repo, bin_ok=False):
    """Download only configuration and weights (.safetensors; .bin only if the repo has no safetensors)."""
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt", "*.model"] + (["pytorch_model.bin"] if bin_ok else [])
    return snapshot_download(repo, allow_patterns=pats)


def _image_features(out):
    """Embedding tensor whether the API returns a tensor or an object with pooler_output."""
    return out if hasattr(out, "shape") else out.pooler_output


def _build_dinov2():
    from transformers import AutoModel
    _threads()
    m = AutoModel.from_pretrained(_hf_path("facebook/dinov2-base")).eval()
    return lambda x: m(pixel_values=x).last_hidden_state[:, 0]          # CLS token (as FD_DINOv2 of Stein 2023)


def _build_dinov3():
    from transformers import AutoModel
    _threads()
    m = AutoModel.from_pretrained(_hf_path("facebook/dinov3-vitb16-pretrain-lvd1689m")).eval()
    return lambda x: m(pixel_values=x).pooler_output


def _build_clip():
    from transformers import CLIPModel
    _threads()
    m = CLIPModel.from_pretrained(_hf_path("openai/clip-vit-base-patch16", bin_ok=True)).eval()
    return lambda x: _image_features(m.get_image_features(pixel_values=x))


def _build_siglip2():
    from transformers import AutoModel
    _threads()
    m = AutoModel.from_pretrained(_hf_path("google/siglip2-base-patch16-224")).eval()
    return lambda x: _image_features(m.get_image_features(pixel_values=x))


# ------------------------------------------------------------------ task evaluator: RT-DETRv2 R50 with 10 % real
RTDETR_REPO = Path(os.environ.get("RTDETR_REPO", paths.KIT.parent / "RT-DETRv2-Densea" / "rtdetrv2_pytorch"))
RTDETR_CFG = Path(os.environ.get("RTDETR_CFG", paths.RUNS_DIR / "configs" / "rtdetrv2" / "px_real10.yml"))
RTDETR_CKPT = Path(os.environ.get("RTDETR_CKPT", paths.RUNS_DIR / "rtdetrv2" / "px_real10" / "best.pth"))
RTDETR_LAYERS = {"enc_p3": 256, "enc_p5": 256, "bb_c5": 2048}     # key -> dimension
RTDETR_BATCH = 10


def roi_mean(fmap, rois):
    """Mean of a map (B,C,H,W) inside the normalized box of each element (cells whose center falls in the box).

    If a box contains no cell center along an axis, the cell closest to its center is used."""
    import torch
    b, _, h, w = fmap.shape
    ys, xs = (torch.arange(h, dtype=torch.float32) + 0.5) / h, (torch.arange(w, dtype=torch.float32) + 0.5) / w
    out = []
    for i in range(b):
        x0, y0, x1, y1 = (float(v) for v in rois[i])
        my, mx = (ys >= y0) & (ys <= y1), (xs >= x0) & (xs <= x1)
        if not my.any():
            my = torch.zeros(h, dtype=torch.bool)
            my[int(min(max((y0 + y1) / 2 * h, 0), h - 1))] = True
        if not mx.any():
            mx = torch.zeros(w, dtype=torch.bool)
            mx[int(min(max((x0 + x1) / 2 * w, 0), w - 1))] = True
        out.append(fmap[i][:, my][:, :, mx].mean(dim=(1, 2)))
    return torch.stack(out)


def _load_rtdetrv2():
    """RT-DETRv2 R50 (px_real10, EMA weights) on CPU with the code of the RT-DETRv2 repo, without a container.

    The /app/RT-DETR and /runs paths of the config are rewritten to the host ones in a temporary yml. `seaborn` is only
    used to draw histograms in `coco_dataset` and may not be installed on the host: it is replaced by an empty module."""
    import tempfile
    import types
    torch = _threads()
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
    sys.path.insert(0, str(RTDETR_REPO))
    from src.core import YAMLConfig
    txt = RTDETR_CFG.read_text(encoding="utf8").replace("/app/RT-DETR", RTDETR_REPO.as_posix())
    txt = txt.replace("output_dir: /runs", "output_dir: " + tempfile.gettempdir().replace("\\", "/"))
    with tempfile.TemporaryDirectory() as d:
        yml = Path(d) / "px_real10_host.yml"
        yml.write_text(txt, encoding="utf8")
        cfg = YAMLConfig(str(yml), PResNet={"pretrained": False})
    ck = torch.load(RTDETR_CKPT, map_location="cpu", weights_only=False)
    model = cfg.model
    model.load_state_dict(ck["ema"]["module"])
    return model.eval()


_RT_MEMO = {}


def _rtdetr_pooled(model, x, rois):
    """{layer: (B,d)} from one backbone+encoder pass, memoized per input tensor (the three layers share the pass)."""
    if _RT_MEMO.get("x") is x:
        return _RT_MEMO["val"]
    import torch
    acc = {k: [] for k in RTDETR_LAYERS}
    for i in range(0, len(x), RTDETR_BATCH):
        bb = model.backbone(x[i:i + RTDETR_BATCH])
        enc = model.encoder(bb)
        r = rois[i:i + RTDETR_BATCH]
        acc["enc_p3"].append(roi_mean(enc[0], r))
        acc["enc_p5"].append(roi_mean(enc[2], r))
        acc["bb_c5"].append(roi_mean(bb[2], r))
    val = {k: torch.cat(v) for k, v in acc.items()}
    _RT_MEMO.update({"x": x, "val": val})
    return val


_RT_MODEL = {}


def _build_rtdetr(layer):
    def fn(x, rois):
        if "m" not in _RT_MODEL:                 # a single model shared by the three layers
            _RT_MODEL["m"] = _load_rtdetrv2()
        return _rtdetr_pooled(_RT_MODEL["m"], x, rois)[layer]
    return fn


_RT_NOTES = {
    "enc_p3": "HybridEncoder P3 (stride 8, 256 d), mean inside the object box; 640x640 input without normalization",
    "enc_p5": "sensitivity: HybridEncoder P5 (stride 32, 256 d), mean inside the box",
    "bb_c5": "sensitivity: ResNet-50-vd backbone C5 (stride 32, 2048 d), mean inside the box"}
RT_NAMES = {"enc_p3": "rtdetrv2_real10", "enc_p5": "rtdetrv2_real10_p5", "bb_c5": "rtdetrv2_real10_c5"}
RT_LIC = "Apache-2.0 (RT-DETR code); own weights: px_real10/best.pth (10 % real TRAIN, EMA)"
_RT_SPECS = [Spec(RT_NAMES[k], "px_real10/best.pth", RT_LIC, ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                  (lambda k=k: _build_rtdetr(k)), _RT_NOTES[k], dim=d, size=640, uses_rois=True)
             for k, d in RTDETR_LAYERS.items()]

REGISTRY = {s.name: s for s in [
    Spec("resnet50", "torchvision IMAGENET1K_V1", "BSD-3-Clause (torchvision code); weights trained on ImageNet-1k",
         IMAGENET, _build_resnet50, "global pooling 2048 d; reference for the AUC of domain_gap", shared_cache=True,
         dim=2048),
    Spec("dinov2_b14", "facebook/dinov2-base", "Apache-2.0 (HF model card)", IMAGENET, _build_dinov2,
         "ViT-B/14, CLS 768 d, 224 squashed input", dim=768),
    Spec("siglip2_b16", "google/siglip2-base-patch16-224", "Apache-2.0 (HF model card)", HALF, _build_siglip2,
         "projected image embedding 768 d", dim=768),
    Spec("dinov3_b16", "facebook/dinov3-vitb16-pretrain-lvd1689m",
         "Meta's own license (DINOv3 License; gated repository)", IMAGENET, _build_dinov3,
         "distilled ViT-B/16, pooler 768 d", gated=True, dim=768),
    Spec("clip_b16", "openai/clip-vit-base-patch16", "MIT (openai/CLIP repo; the HF model card declares no license)",
         CLIP_MS, _build_clip, "projected image embedding 512 d (the one used by CMMD)", dim=512),
] + _RT_SPECS}
TASK = ["rtdetrv2_real10"]      # task evaluator; apart from the 5 general extractors of the panel
TASK_SENS = ["rtdetrv2_real10_p5", "rtdetrv2_real10_c5"]    # alternative layers (same pass; sensitivity only)
ORDER = ["resnet50", "dinov2_b14", "siglip2_b16", "dinov3_b16", "clip_b16"]   # priority if the CPU is scarce


@contextlib.contextmanager
def cache_at(path):
    """Temporarily point the feature cache to `path` (the panel scripts keep one cache per scenario)."""
    global CACHE
    old, CACHE = CACHE, Path(path)
    try:
        yield
    finally:
        CACHE = old


def available(names=None):
    """(available, failures): loads each model once; a failure (access, network) is reported and does not stop the panel."""
    ok, bad = [], {}
    for n in names or ORDER:
        try:
            REGISTRY[n].fn()
            ok.append(n)
        except Exception as e:  # noqa: BLE001
            bad[n] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return ok, bad


def items_hash(items):
    return hashlib.sha1(json.dumps([[i.get("path"), i.get("bbox")] for i in items]).encode()).hexdigest()


def _squash(item, size=SIZE):
    from PIL import Image
    return dg.load_pil(item, "obj").resize((size, size), Image.BILINEAR)   # like T.Resize of domain_gap


def box_in_crop(item):
    """Object box inside the crop with context, normalized to [0,1] -> [x0, y0, x1, y1]."""
    x, y, bw, bh = item["bbox"]
    cx0, cy0, cx1, cy1 = dg.expand_box(item["bbox"], item["w"], item["h"])
    sw, sh = max(cx1 - cx0, 1), max(cy1 - cy0, 1)
    r = [(x - cx0) / sw, (y - cy0) / sh, (x + bw - cx0) / sw, (y + bh - cy0) / sh]
    return [min(max(v, 0.0), 1.0) for v in r]


def _tensor(pils, mean_std):
    import torch
    a = np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in pils])
    m, s = (np.asarray(v, dtype=np.float32).reshape(1, 1, 1, 3) for v in mean_std)
    return torch.from_numpy(((a - m) / s).transpose(0, 3, 1, 2).copy())


def _shared_array(setname, n_rows):
    """Earlier ResNet-50 cache of the set (if it exists and has the expected rows)."""
    for f in (dg.CACHE / f"{setname}.npy", dg.CACHE / "variant_pilot" / f"{setname}_obj.npy",
              dg.CACHE / "external_seabed" / f"{setname}_obj.npy"):
        if f.exists():
            a = np.load(f)
            if a.shape[0] == n_rows:
                return a
    return None


def _save(n, setname, arr, h):
    (CACHE / n).mkdir(parents=True, exist_ok=True)
    np.save(CACHE / n / f"{setname}.npy", arr.astype(np.float32))
    (CACHE / n / f"{setname}.sha1").write_text(h)


def extract_set(setname, items, names, shared_name=None, chunk=50, log=print):
    """Features of `items` for each extractor of `names` -> {name: (n, d)}. Resumable.

    `shared_name`: name of the earlier ResNet-50 cache (e.g. 'real_train_obj', 'synth_obj' or the variant)."""
    h, out, todo = items_hash(items), {}, []
    for n in names:
        f, hf = CACHE / n / f"{setname}.npy", CACHE / n / f"{setname}.sha1"
        shared = _shared_array(shared_name, len(items)) if REGISTRY[n].shared_cache and shared_name else None
        if f.exists() and hf.exists() and hf.read_text().strip() == h:
            out[n] = np.load(f)
        elif shared is not None:
            _save(n, setname, shared, h)
            out[n] = shared
        else:
            todo.append(n)
    if todo:
        out.update(_extract_chunks(setname, items, todo, h, chunk, log))
    return out


def _run_chunk(chunk_items, need):
    """Pass a chunk of crops through the extractors `need` -> {name: (n, d)}. PIL images and tensors are shared between
    the extractors with the same size and normalization (the three RT-DETRv2 layers also share the model pass)."""
    import torch
    pils, tensors, rois, out = {}, {}, None, {}
    for n in need:
        sp = REGISTRY[n]
        if sp.size not in pils:
            pils[sp.size] = [_squash(it, sp.size) for it in chunk_items]
        key = (sp.size, tuple(map(tuple, sp.mean_std)))
        if key not in tensors:
            tensors[key] = _tensor(pils[sp.size], sp.mean_std)
        args = (tensors[key],)
        if sp.uses_rois:
            rois = rois if rois is not None else torch.tensor([box_in_crop(it) for it in chunk_items])
            args += (rois,)
        with torch.no_grad():
            out[n] = sp.fn()(*args).numpy().astype(np.float32)
    return out


def _extract_chunks(setname, items, names, h, chunk, log):
    t0, parts = time.time(), {n: [] for n in names}
    for n in names:
        d = CACHE / n / setname
        d.mkdir(parents=True, exist_ok=True)
        if (d / "items.sha1").exists() and (d / "items.sha1").read_text().strip() != h:
            for c in d.glob("c*.npy"):
                c.unlink()
        (d / "items.sha1").write_text(h)
    for k in range(0, len(items), chunk):
        need = [n for n in names if not (CACHE / n / setname / f"c{k:05d}.npy").exists()]
        if need:
            for n, e in _run_chunk(items[k:k + chunk], need).items():
                np.save(CACHE / n / setname / f"c{k:05d}.npy", e)
        for n in names:
            parts[n].append(np.load(CACHE / n / setname / f"c{k:05d}.npy"))
        log(f"[{setname}] {min(k + chunk, len(items))}/{len(items)} {time.time() - t0:.0f}s")
    res = {n: np.concatenate(p) for n, p in parts.items()}
    for n, a in res.items():
        _save(n, setname, a, h)
    return res
