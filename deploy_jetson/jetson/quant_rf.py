"""INT8 quantization with Q/DQ nodes (ModelOpt 0.19, CPU) of the RF-DETR ONNX model, run on the Jetson. TRAIN images
are the only calibration data, never VAL or TEST.

--calib random      normalized random tensors, ONLY to measure latency and power (the AP is meaningless);
--calib DIR         calibration images (the TRAIN list of make_calibration_set.py copied with --copy-to),
                    preprocessed as in VAL (eval_rfdetr_val.prepare);
--exclude           comma-separated regexes of nodes that stay in FP16;
--only              comma-separated regexes: ONLY these nodes are quantized;
--op_types          operator types to quantize (empty = the ModelOpt defaults, which also quantize Add, Mul, Slice
                    and Shape and make the RF-DETR AP collapse);
--disable_mha_qdq   keep the attention matmuls (Q.K and A.V) in FP16.

Usage: venv_mo/bin/python quant_rf.py --onnx models/rfdetr_exp07.onnx --out models/rfdetr_exp07_qG.onnx \
           --calib calib/train_100 --n 100 --method entropy --op_types MatMul,Gemm,Conv --disable_mha_qdq \
           --only REGEX[,REGEX...]
"""
import argparse
import glob
import os

import numpy as np


def calib_data(spec, n):
    if spec == "random":
        return np.random.default_rng(0).standard_normal((n, 3, 560, 560)).astype(np.float32)
    from PIL import Image
    from eval_rfdetr_val import prepare
    files = sorted(glob.glob(os.path.join(spec, "*.jpg")) + glob.glob(os.path.join(spec, "*.png")))[:n]
    return np.concatenate([prepare(Image.open(f).convert("RGB")) for f in files])


def split(arg):
    return [r for r in arg.split(",") if r]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="random")
    ap.add_argument("--n", type=int, default=16, help="number of calibration tensors / images")
    ap.add_argument("--method", default="max", choices=["max", "entropy"])
    ap.add_argument("--exclude", default="")
    ap.add_argument("--op_types", default="", help="for example MatMul,Gemm,Conv (empty = ModelOpt defaults)")
    ap.add_argument("--only", default="")
    ap.add_argument("--disable_mha_qdq", action="store_true")
    a = ap.parse_args()
    from modelopt.onnx.quantization import quantize
    data = calib_data(a.calib, a.n)
    print("calibration", data.shape, flush=True)
    quantize(onnx_path=a.onnx, quantize_mode="int8", calibration_data=data, calibration_method=a.method,
             calibration_eps=["cpu"], nodes_to_exclude=split(a.exclude) or None,
             nodes_to_quantize=split(a.only) or None, op_types_to_quantize=split(a.op_types) or None,
             disable_mha_qdq=a.disable_mha_qdq, output_path=a.out, use_external_data_format=False, verbose=False)
    import resource   # Linux only
    print("OK", a.out, "peak RAM", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "MB", flush=True)


if __name__ == "__main__":
    main()
