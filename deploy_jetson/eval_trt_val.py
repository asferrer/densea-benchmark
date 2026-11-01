"""VAL AP of an RT-DETRv2 TensorRT engine (FP16 or mixed INT8) on the Jetson. VAL only, never TEST.

Same logic as ``eval_onnx_val.py`` (``rtdetrv2_io.preprocess``, ``orig_target_sizes = [width, height]``,
COCO detections with score > 0.001) with the engine instead of onnxruntime. It also times, per image, the CPU
preprocessing and the synchronized GPU inference and stores them in ``<out>_ap.json``. The reference latency is the
``trtexec`` one; these are the times of the whole Python loop.

On the Jetson run it with the Python that has torch and pycocotools and add the system TensorRT bindings:
``PYTHONPATH=/usr/lib/python3.10/dist-packages python3 eval_trt_val.py ...``

Usage:
    python eval_trt_val.py --engine m.engine --val data/val.json --data_root data --out results/preds.json [--n 0]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import coco_ap, time_stats  # noqa: E402
from rtdetrv2_io import detections, preprocess  # noqa: E402

class Engine:
    """Deserialized engine with one preallocated CUDA buffer per I/O tensor."""

    def __init__(self, path):
        import tensorrt as trt
        import torch
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                  trt.int64: torch.int64, trt.bool: torch.bool}
        logger = trt.Logger(trt.Logger.WARNING)
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

    def __call__(self, images, sizes):
        self.buf["images"].copy_(self.torch.from_numpy(images))
        self.buf["orig_target_sizes"].copy_(self.torch.from_numpy(sizes).to(self.buf["orig_target_sizes"].dtype))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return [self.buf[k].cpu().numpy() for k in ("labels", "boxes", "scores")]


def predict(eng, images, data_root, n):
    images = sorted(images, key=lambda d: d["id"])[: n or None]
    preds, t_pre, t_inf = [], [], []
    for k, im in enumerate(images):
        bgr = cv2.imread(os.path.join(data_root, im["file_name"]))
        t0 = time.perf_counter()
        x = preprocess(bgr)
        t1 = time.perf_counter()
        labels, boxes, scores = eng(x, np.array([[bgr.shape[1], bgr.shape[0]]], dtype=np.int32))
        t2 = time.perf_counter()
        t_pre.append(t1 - t0)
        t_inf.append(t2 - t1)
        if scores.ndim == 2:   # outputs with a batch dimension; the exported model has none ([300])
            labels, boxes, scores = labels[0], boxes[0], scores[0]
        preds += detections(im["id"], labels, boxes, scores)
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
    a = ap.parse_args()
    with open(a.val, encoding="utf8") as f:
        val = json.load(f)
    preds, ids, t_pre, t_inf = predict(Engine(a.engine), val["images"], a.data_root, a.n)
    with open(a.out, "w") as f:
        json.dump(preds, f)
    res = coco_ap(a.val, preds, ids)
    with open(a.out.replace(".json", "_ap.json"), "w") as f:
        json.dump({"engine": a.engine, "n_img": len(ids), **res, "preprocess_ms": time_stats(t_pre),
                   "inference_ms": time_stats(t_inf)}, f, indent=1)
    print({k: round(100 * v, 2) for k, v in res.items()}, "inference", time_stats(t_inf))


if __name__ == "__main__":
    main()
