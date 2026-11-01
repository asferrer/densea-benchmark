"""VAL AP of an RT-DETRv2 ONNX model (FP32 or INT8 with Q/DQ nodes) with onnxruntime on the CPU. VAL only, never TEST.

Estimates on the PC the loss caused by quantization before the engine is built on the Jetson: onnxruntime runs the
Q/DQ nodes simulating INT8, so the AP approximates (does not equal) the AP of the TensorRT engine.
Preprocessing is ``rtdetrv2_io.preprocess`` with ``orig_target_sizes = [width, height]``. Predictions are
COCO detections with score > 0.001.

Usage:
    python deploy_jetson/eval_onnx_val.py --onnx model.onnx --out preds.json [--n 0] [--threads 8]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from eval_utils import coco_ap  # noqa: E402
from rtdetrv2_io import detections, preprocess  # noqa: E402


def session(onnx_path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


def predict(sess, images, data_root, n):
    images = sorted(images, key=lambda d: d["id"])[: n or None]
    preds, t0 = [], time.time()
    for k, im in enumerate(images):
        bgr = cv2.imread(os.path.join(data_root, im["file_name"]))
        size = np.array([[bgr.shape[1], bgr.shape[0]]], dtype=np.int32)
        labels, boxes, scores = sess.run(None, {"images": preprocess(bgr), "orig_target_sizes": size})
        if scores.ndim == 2:   # outputs with a batch dimension; the exported model has none ([300])
            labels, boxes, scores = labels[0], boxes[0], scores[0]
        preds += detections(im["id"], labels, boxes, scores)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(images)} {(time.time() - t0) / (k + 1):.2f} s/img", flush=True)
    return preds, [im["id"] for im in images]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="predictions JSON; the AP goes to <out>_ap.json")
    ap.add_argument("--data_root", default=str(paths.DATA_ROOT))
    ap.add_argument("--n", type=int, default=0, help="0 = all of VAL")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    with open(paths.annotations_path("val"), encoding="utf8") as f:
        val = json.load(f)
    preds, ids = predict(session(a.onnx, a.threads), val["images"], a.data_root, a.n)
    with open(a.out, "w") as f:
        json.dump(preds, f)
    res = coco_ap(paths.annotations_path("val"), preds, ids)
    with open(a.out.replace(".json", "_ap.json"), "w") as f:
        json.dump({"onnx": a.onnx, "n_img": len(ids), **res}, f, indent=1)
    print({k: round(100 * v, 2) for k, v in res.items()})


if __name__ == "__main__":
    main()
