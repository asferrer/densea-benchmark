"""AP on the first N VAL images (by id) of ONNX models run with onnxruntime on the CPU, and of existing predictions
(for example from TensorRT) cut to the same images. It separates the error of the quantization (Q/DQ under
onnxruntime) from the error of TensorRT running the Q/DQ graph. VAL only. RF-DETR models.

Usage: python ap_ort_subset.py --val data/val.json --data_root data --n 100 \
           --onnx models/a.onnx [models/b.onnx ...] --preds results/x.json [...]
"""
import argparse
import json

import onnxruntime as ort

from eval_rfdetr_val import predict
from eval_utils import coco_ap


class OrtEngine:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 6
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        dets, labels = self.s.run(["dets", "labels"], {"input": x})
        return dets[0], labels[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--onnx", nargs="*", default=[])
    ap.add_argument("--preds", nargs="*", default=[])
    a = ap.parse_args()
    with open(a.val, encoding="utf8") as f:
        val = json.load(f)
    cats = {c["id"] for c in val["categories"]}
    ids = [im["id"] for im in sorted(val["images"], key=lambda d: d["id"])[: a.n]]
    res = {}
    for p in a.onnx:
        preds, _, _, _ = predict(OrtEngine(p), val["images"], a.data_root, a.n, 300, cats)
        res[p] = coco_ap(a.val, preds, ids)
    keep = set(ids)
    for p in a.preds:
        with open(p) as f:
            res[p] = coco_ap(a.val, [d for d in json.load(f) if d["image_id"] in keep], ids)
    for p, r in res.items():
        print(f"{p}: AP {100 * r['AP']:.2f}  AP50 {100 * r['AP50']:.2f}", flush=True)


if __name__ == "__main__":
    main()
