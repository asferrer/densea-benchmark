"""VAL AP of an RF-DETR TensorRT engine on the Jetson, plus the pre/post-processing shared with the ONNX export.
VAL only, never TEST.

Preprocessing and post-processing follow the RF-DETR library:
  - VAL transform: ``SquareResize([560])`` (PIL bilinear to 560 x 560, aspect ratio not kept), ``ToTensor`` and
    ``Normalize`` with the ImageNet mean and standard deviation;
  - ``PostProcess``: sigmoid, top-``num_select`` over queries x classes, cxcywh boxes to xyxy scaled to the original
    size. ``num_select`` is 300, as in the reference predictions computed on the PC.
The head has one output more than classes (9 against 8). Its detections have no ground truth and COCOeval ignores
them, so they are dropped from the JSON (the AP does not change) and the bootstrap does not reject the file.
``export_onnx_rfdetr.py`` imports ``prepare`` and ``postprocess`` from here so that the PC and the Jetson share code.
Per image it times the CPU preprocessing and the synchronized GPU inference; the reference latency is ``trtexec``.

On the Jetson: ``PYTHONPATH=/usr/lib/python3.10/dist-packages python3 eval_rfdetr_val.py ...``

Usage:
    python eval_rfdetr_val.py --engine m.engine --val data/val.json --data_root data --out results/preds.json [--n 0]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import coco_ap, time_stats  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare(img, res=560):
    """PIL RGB image -> contiguous float32 tensor (1, 3, res, res); TensorRT needs contiguous inputs."""
    x = np.asarray(img.resize((res, res), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(dets, logits, orig_wh, num_select=300):
    """dets (Q, 4) normalized cxcywh and logits (Q, C) -> scores, labels and xyxy boxes in original pixels."""
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = prob.reshape(-1)
    idx = np.argpartition(-flat, num_select - 1)[:num_select]
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    q, labels = idx // logits.shape[1], idx % logits.shape[1]
    cx, cy, w, h = dets[q].astype(np.float64).T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * np.tile(orig_wh, 2)
    return flat[idx], labels, boxes


class Engine:
    """Deserialized engine with one preallocated CUDA buffer per I/O tensor."""

    def __init__(self, path):
        import tensorrt as trt
        import torch
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                  trt.int64: torch.int64}
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.torch = torch
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.buf[name] = torch.empty(shape, dtype=dtypes[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, self.buf[name].data_ptr())

    def __call__(self, x):
        self.buf["input"].copy_(self.torch.from_numpy(x))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buf["dets"][0].float().cpu().numpy(), self.buf["labels"][0].float().cpu().numpy()


def predict(eng, images, data_root, n, num_select, cats):
    """COCO detections of the first ``n`` images by id (all if 0) and per-image preprocessing / inference times."""
    images = sorted(images, key=lambda d: d["id"])[: n or None]
    preds, t_pre, t_inf = [], [], []
    for k, im in enumerate(images):
        img = Image.open(os.path.join(data_root, im["file_name"])).convert("RGB")
        t0 = time.perf_counter()
        x = prepare(img)
        t1 = time.perf_counter()
        dets, logits = eng(x)
        t2 = time.perf_counter()
        t_pre.append(t1 - t0)
        t_inf.append(t2 - t1)
        scores, labels, boxes = postprocess(dets, logits, np.array(img.size, dtype=np.float64), num_select)
        for lab, b, s in zip(labels, boxes, scores):
            if int(lab) in cats:
                preds.append({"image_id": im["id"], "category_id": int(lab),
                              "bbox": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2] - b[0]), 2),
                                       round(float(b[3] - b[1]), 2)], "score": round(float(s), 6)})
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(images)} inference {1000 * np.mean(t_inf):.1f} ms", flush=True)
    return preds, [im["id"] for im in images], t_pre, t_inf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--val", required=True, help="VAL annotations (COCO)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True, help="predictions JSON; the AP goes to <out>_ap.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all of VAL")
    ap.add_argument("--num_select", type=int, default=300)
    a = ap.parse_args()
    with open(a.val, encoding="utf8") as f:
        val = json.load(f)
    preds, ids, t_pre, t_inf = predict(Engine(a.engine), val["images"], a.data_root, a.n, a.num_select,
                                       {c["id"] for c in val["categories"]})
    with open(a.out, "w") as f:
        json.dump(preds, f, separators=(",", ":"))
    res = coco_ap(a.val, preds, ids)
    with open(a.out.replace(".json", "_ap.json"), "w") as f:
        json.dump({"engine": a.engine, "n_img": len(ids), **res, "preprocess_ms": time_stats(t_pre),
                   "inference_ms": time_stats(t_inf)}, f, indent=1)
    print({k: round(100 * v, 2) for k, v in res.items()}, "inference", time_stats(t_inf))


if __name__ == "__main__":
    main()
