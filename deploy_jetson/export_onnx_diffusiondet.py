"""Export DiffusionDet (``SAMPLE_STEP 1``) to ONNX for TensorRT and check the equivalence. VAL only, never TEST.

With a single sampling step, inference is one pass (backbone + FPN, then the six heads over ``NUM_PROPOSALS`` noise
boxes) followed by a top-k. The engine takes the noise as an input, so PyTorch, ONNX and TensorRT can all receive
the same one.
  inputs:  ``image`` (1, 3, H, W) RGB 0-255, already resized (short side 800, long side <= 960) and padded to
           multiples of 32 with the pixel mean (after normalization the padding is 0, as in the detectron2
           ``ImageList``); ``images_whwh`` (1, 4) = [w, h, w, h] of the resized image without padding;
           ``noise`` (1, P, 4) ~ N(0, 1).
  outputs: ``logits`` (1, P, C) and ``boxes`` (1, P, 4) of the last head (absolute xyxy in the resized image).
Outside the engine (``postprocess``): sigmoid, top-P over P x C scores, per-class NMS at IoU 0.5 (``USE_NMS`` is on
by default in DiffusionDet), rescaling to the original size, clipping and removal of empty boxes, as in
``DiffusionDet.inference`` + ``detector_postprocess``.

Requirements of the export:
  - The image fed to TensorRT must be a contiguous NCHW array (``prepare`` returns one). TensorRT reads the pointer
    as dense NCHW, so a transposed view silently produces unrelated outputs.
  - The model is traced with a 640 x 960 dummy image and exported with dynamic ``H`` and ``W`` axes. The TensorRT
    engines are built with the profile 544x800 (min) / 800x800 (opt) / 800x960 (max).
  - ``trtexec`` with random inputs does not measure the latency of this model, because the boxes decide the pyramid
    level of each RoI. Use real inputs (``jetson/dump_inputs.py`` and ``trtexec --loadInputs``).

Checks with ``--check N`` (the first N VAL images by id, noise from ``torch.manual_seed(image id)``):
  1) PyTorch wrapper against the original model (same noise): identical detections;
  2) ONNX (onnxruntime, CPU) against the wrapper: maximum logit and box difference.

Run it on the CPU inside the DiffusionDet container (``DD_CODE_DIR`` is the folder with ``diffusiondet`` and the
detectron2 configs; ``DD_CONFIG`` the YAML configuration):
    python /kit/deploy_jetson/export_onnx_diffusiondet.py --weights /runs/deploy_jetson/dd/<name>_ema.pth \
        --out /runs/deploy_jetson/models/diffusiondet_<name>.onnx --check 3
The weights file is the ``ema_state`` of ``model_best.pth`` saved under the key ``model``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

KIT = Path(__file__).resolve().parents[1]
CFG = os.environ.get("DD_CONFIG", "/app/densea-diffusiondet/configs/diffdet.densea.res50.yaml")
sys.path.insert(0, os.environ.get("DD_CODE_DIR", "/app/densea-diffusiondet"))


class DDCore(nn.Module):
    """Backbone + FPN + one-step heads of DiffusionDet with the noise as an input."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("mean", torch.tensor(model.cfg_pixel_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(model.cfg_pixel_std).view(1, 3, 1, 1))
        self.t = model.num_timesteps - 1   # with SAMPLE_STEP 1, time_pairs = [(T - 1, -1)]

    def forward(self, image, images_whwh, noise):
        x = (image - self.mean) / self.std
        src = self.m.backbone(x)
        feats = [src[f] for f in self.m.in_features]
        t = torch.full((1,), self.t, dtype=torch.long, device=image.device)
        _, cls, coord = self.m.model_predictions(feats, images_whwh, noise, t)
        return cls[-1], coord[-1]


def get_config(weights="", device="cpu"):
    from detectron2.config import get_cfg
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.set_new_allowed(True)   # training-only keys of the YAML are accepted and ignored
    cfg.merge_from_file(CFG)
    cfg.MODEL.WEIGHTS, cfg.MODEL_EMA.ENABLED, cfg.MODEL.DEVICE = weights, False, device
    return cfg


def load(weights, device="cpu"):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    cfg = get_config(weights, device)
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(weights)
    model.cfg_pixel_mean, model.cfg_pixel_std = list(cfg.MODEL.PIXEL_MEAN), list(cfg.MODEL.PIXEL_STD)
    assert cfg.MODEL.DiffusionDet.SAMPLE_STEP == 1, "the wrapper is only valid with SAMPLE_STEP 1"
    return cfg, model


def prepare(img_rgb, cfg, size_div=32):
    """detectron2 resize (ResizeShortestEdge) and padding with the pixel mean up to multiples of 32.

    Returns the contiguous NCHW image, ``images_whwh`` and the resized (height, width).
    """
    import detectron2.data.transforms as T
    aug = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    x = aug.get_transform(img_rgb).apply_image(img_rgb).astype(np.float32)
    h, w = x.shape[:2]
    H, W = -(-h // size_div) * size_div, -(-w // size_div) * size_div
    pad = np.empty((H, W, 3), np.float32)
    pad[:] = np.array(cfg.MODEL.PIXEL_MEAN, np.float32)
    pad[:h, :w] = x
    return np.ascontiguousarray(pad.transpose(2, 0, 1)[None]), np.array([[w, h, w, h]], np.float32), (h, w)


def postprocess(logits, boxes, resized_hw, orig_hw, k, nms_iou=0.5):
    """Sigmoid, top-k over P x C, per-class NMS, rescaling to the original size, clipping, non-empty boxes."""
    from torchvision.ops import batched_nms
    lg, bx = torch.as_tensor(logits[0]), torch.as_tensor(boxes[0])
    n_cls = lg.shape[-1]
    s, idx = torch.sigmoid(lg).flatten().topk(k, sorted=False)
    lab, b = idx % n_cls, bx[idx // n_cls]
    keep = batched_nms(b, s, lab, nms_iou)
    b, s, lab = b[keep].clone(), s[keep], lab[keep]
    (h, w), (H0, W0) = resized_hw, orig_hw
    b[:, 0::2] = (b[:, 0::2] * (W0 / w)).clamp(0, W0)
    b[:, 1::2] = (b[:, 1::2] * (H0 / h)).clamp(0, H0)
    ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[ok].numpy(), s[ok].numpy(), lab[ok].numpy()


def check_image(im, cfg, model, core, sess):
    import cv2
    import detectron2.data.transforms as T
    from common import paths
    n_prop = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
    img = cv2.imread(str(paths.DATA_ROOT / im["file_name"]))[:, :, ::-1].copy()
    image, whwh, rhw = prepare(img, cfg)
    torch.manual_seed(im["id"])
    noise = torch.randn(1, n_prop, 4)
    with torch.no_grad():
        lg, bx = core(torch.from_numpy(image), torch.from_numpy(whwh), noise)
        torch.manual_seed(im["id"])   # the original draws the same noise with its first torch.randn
        aug = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        x = aug.get_transform(img).apply_image(img)
        ref = model([{"image": torch.as_tensor(x.astype("float32").transpose(2, 0, 1)),
                      "height": img.shape[0], "width": img.shape[1]}])[0]["instances"]
    b, s, _ = postprocess(lg.numpy(), bx.numpy(), rhw, img.shape[:2], n_prop)
    o, r = np.argsort(-s), ref.scores.argsort(descending=True)
    db = np.abs(b[o] - ref.pred_boxes.tensor[r].numpy()).max()
    ds = np.abs(s[o] - ref.scores[r].numpy()).max()
    msg = f"id {im['id']} {img.shape[:2]} -> {image.shape[-2:]}: wrapper vs original |boxes| {db:.4f}, |score| {ds:.2e}"
    if sess:
        ol, ob = sess.run(None, {"image": image, "images_whwh": whwh, "noise": noise.numpy()})
        msg += (f"; ONNX vs wrapper |logits| {np.abs(ol - lg.numpy()).max():.2e}, "
                f"|boxes| {np.abs(ob - bx.numpy()).max():.2e}")
    print(msg, flush=True)


def check(cfg, model, core, onnx_path, n):
    import onnxruntime as ort
    from common import paths
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]) if onnx_path else None
    with open(paths.annotations_path("val"), encoding="utf8") as f:
        ims = sorted(json.load(f)["images"], key=lambda d: d["id"])[:n]
    for im in ims:
        check_image(im, cfg, model, core, sess)


def main():
    sys.path.insert(0, str(KIT))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--skip-export", action="store_true", help="only run the checks on an existing ONNX file")
    a = ap.parse_args()
    cfg, model = load(a.weights)
    core = DDCore(model).eval()
    n_prop = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
    if not a.skip_export:
        dummy = (torch.full((1, 3, 640, 960), 120.0), torch.tensor([[960., 620., 960., 620.]]),
                 torch.randn(1, n_prop, 4))
        with torch.no_grad():
            torch.onnx.export(core, dummy, a.out, opset_version=a.opset,
                              input_names=["image", "images_whwh", "noise"], output_names=["logits", "boxes"],
                              dynamic_axes={"image": {2: "H", 3: "W"}}, do_constant_folding=True, dynamo=False)
        print("ONNX:", a.out, os.path.getsize(a.out) // 2**20, "MB", flush=True)
    if a.check:
        check(cfg, model, core, None if a.skip_export and not os.path.exists(a.out) else a.out, a.check)


if __name__ == "__main__":
    main()
