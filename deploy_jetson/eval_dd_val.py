"""VAL AP and per-image time of DiffusionDet on the Jetson: TensorRT engine or PyTorch wrapper. VAL only, never TEST.

Both backends receive the SAME noise (``torch.manual_seed(image id)``, drawn on the CPU), so their predictions can
be compared pair by pair. Pre/post-processing is the one of ``export_onnx_diffusiondet.py`` (detectron2
ResizeShortestEdge, padding with the mean to multiples of 32; sigmoid, top-k, per-class NMS at 0.5, rescaling,
non-empty boxes). Predictions are COCO detections without a score threshold (everything that survives the NMS).

Environment (``jetson/dd_bench.sh`` sets it): ``DD_CODE_DIR`` folder with the ``diffusiondet`` package, ``DD_CONFIG``
YAML configuration, ``LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6`` and
``PYTHONPATH=/usr/lib/python3.10/dist-packages`` (system TensorRT bindings), run with the DiffusionDet virtualenv.

Usage:
    python eval_dd_val.py --backend trt --engine m.engine --val data/val.json --data_root data --out results/x.json
    python eval_dd_val.py --backend torch --weights m.pth --val data/val.json --data_root data --out results/x.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_onnx_diffusiondet as ex  # noqa: E402
from eval_utils import coco_ap, time_stats  # noqa: E402


class TRT:
    """TensorRT engine with a dynamic image shape."""

    def __init__(self, path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")   # ROIAlign_TRT (ONNX RoiAlign); trtexec loads the plugins itself
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"could not load {path}"
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, image, whwh, noise):
        ins = {"image": torch.from_numpy(image).cuda().contiguous(), "images_whwh": torch.from_numpy(whwh).cuda(),
               "noise": noise.cuda().contiguous()}
        self.ctx.set_input_shape("image", tuple(image.shape))
        outs = {n: torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
                for n in ("logits", "boxes")}
        for n, t in {**ins, **outs}.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outs["logits"].cpu().numpy(), outs["boxes"].cpu().numpy()


class Torch:
    """PyTorch wrapper (``DDCore``) on the GPU."""

    def __init__(self, weights):
        _, model = ex.load(weights, device="cuda")
        self.core = ex.DDCore(model).cuda().eval()

    @torch.no_grad()
    def __call__(self, image, whwh, noise):
        lg, bx = self.core(torch.from_numpy(image).cuda(), torch.from_numpy(whwh).cuda(), noise.cuda())
        torch.cuda.synchronize()
        return lg.cpu().numpy(), bx.cpu().numpy()


def predict(net, cfg, images, data_root, cat_ids):
    """COCO detections of the images (given order) and per-image preprocessing and network times in seconds."""
    n_prop = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
    preds, t_pre, t_net = [], [], []
    for k, im in enumerate(images):
        img = cv2.imread(os.path.join(data_root, im["file_name"]))[:, :, ::-1].copy()
        t0 = time.perf_counter()
        image, whwh, rhw = ex.prepare(img, cfg)
        torch.manual_seed(im["id"])
        noise = torch.randn(1, n_prop, 4)
        t1 = time.perf_counter()
        lg, bx = net(image, whwh, noise)
        t2 = time.perf_counter()
        b, s, lab = ex.postprocess(lg, bx, rhw, img.shape[:2], n_prop)
        t_pre.append(t1 - t0)
        t_net.append(t2 - t1)
        for bb, ss, ll in zip(b, s, lab):
            preds.append({"image_id": im["id"], "category_id": cat_ids[int(ll)],
                          "bbox": [float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                          "score": float(ss)})
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(images)} network {1000 * np.mean(t_net[10:]):.1f} ms", flush=True)
    return preds, t_pre, t_net


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True, choices=["trt", "torch"])
    ap.add_argument("--engine", default="")
    ap.add_argument("--weights", default="", help="EMA weights (.pth) for the engine config and the torch backend")
    ap.add_argument("--val", required=True, help="VAL annotations (COCO)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True, help="predictions JSON; the AP goes to <out>_ap.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all of VAL")
    a = ap.parse_args()
    if (a.backend == "trt" and not a.engine) or (a.backend == "torch" and not a.weights):
        ap.error("--engine is required with --backend trt and --weights with --backend torch")
    cfg = ex.get_config(a.weights)
    net = TRT(a.engine) if a.backend == "trt" else Torch(a.weights)
    with open(a.val, encoding="utf8") as f:
        val = json.load(f)
    cat_ids = sorted(c["id"] for c in val["categories"])   # detectron2: contiguous index i = i-th sorted id
    ims = sorted(val["images"], key=lambda d: d["id"])[: a.n or None]
    preds, t_pre, t_net = predict(net, cfg, ims, a.data_root, cat_ids)
    with open(a.out, "w") as f:
        json.dump(preds, f)
    res = {"backend": a.backend, "engine": a.engine, "n_img": len(ims),
           **coco_ap(a.val, preds, [im["id"] for im in ims]),
           "preprocess_ms": time_stats(t_pre, 10), "net_ms": time_stats(t_net, 10)}
    with open(a.out.replace(".json", "_ap.json"), "w") as f:
        json.dump(res, f, indent=1)
    print({k: (round(100 * v, 2) if k.startswith("AP") else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
