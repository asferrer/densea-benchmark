"""INT8 quantization (Q/DQ nodes) of the RT-DETRv2 ONNX model with NVIDIA ModelOpt, calibrated on TRAIN images.

Quantize on the PC (CPU) and build the engine on the Jetson with ``jetson/build_engines.sh``: the Q/DQ ONNX does not
depend on the hardware, and the calibration loads every image in memory (1,000 images of 640 x 640 in float32 are
about 4.7 GB, too much for the 8 GB shared memory of the Orin Nano).

Calibration images come from a list written by ``make_calibration_set.py`` (``--calib_list``): the first
``--max_images`` entries, in order, resolved to files through the TRAIN annotations of the group partition.
Preprocessing is ``rtdetrv2_io.preprocess`` (the one used at deployment time).

Only the backbone and the convolutional part of the encoder are quantized: the decoder, the post-processor and the
score / box heads are excluded because deformable attention and the sigmoid collapse the scores when quantized.
``--exclude_extra`` keeps more nodes in FP16. The configuration of the paper keeps the three input convolutions out
(``--exclude_extra '.*backbone/conv1/.*'``).

Requirements: ``pip install nvidia-modelopt==0.19.0 --no-deps`` plus onnx-graphsurgeon, pydantic, pulp, ml_dtypes,
cppimport, ninja, regex, rich, safetensors (Linux or WSL; the C++ extension of ModelOpt does not build on Windows).

Usage:
    python deploy_jetson/quantize_int8_rtdetrv2.py --onnx model_simplified.onnx \
        --calib_list deploy_jetson/calibration/calib_100.json --calibrate_method entropy \
        --exclude_extra '.*backbone/conv1/.*' --output_onnx model_int8mix_stem.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from rtdetrv2_io import preprocess  # noqa: E402

NODES_TO_EXCLUDE = [
    ".*decoder.*",
    ".*postprocessor.*",
    ".*enc_score_head.*",
    ".*enc_bbox_head.*",
    ".*dec_score_head.*",
    ".*dec_bbox_head.*",
    ".*query_pos_head.*",
]
METHOD_MAP = {"entropy": "entropy", "minmax": "max"}   # ModelOpt calls MinMax "max"


def calibration_files(calib_list, data_root, max_images=0):
    """Image files of the first ``max_images`` entries (all if 0) of a TRAIN calibration list."""
    with open(calib_list, encoding="utf8") as f:
        meta = json.load(f)
    if meta.get("split") != "train":
        sys.exit(f"{calib_list} is not a TRAIN list (split={meta.get('split')})")
    rows = meta["images"]
    if max_images > len(rows):
        sys.exit(f"{calib_list} has {len(rows)} images, {max_images} requested")
    with open(paths.annotations_path("train"), encoding="utf8") as f:
        file_of = {im["id"]: im["file_name"] for im in json.load(f)["images"]}
    return [Path(data_root) / file_of[r["id"]] for r in (rows[:max_images] if max_images else rows)]


def prepare_calibration_data(files):
    """Preprocessed images (N, 3, 640, 640) and original sizes [width, height] (N, 2) for the two model inputs."""
    images, sizes = [], []
    for k, path in enumerate(files, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            sys.exit(f"could not read {path}")
        images.append(preprocess(bgr))
        sizes.append(np.array([[bgr.shape[1], bgr.shape[0]]], dtype=np.int32))
        if k % 10 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] preprocessed")
    data = {"images": np.concatenate(images), "orig_target_sizes": np.concatenate(sizes)}
    print(f"calibration images {data['images'].shape} ({data['images'].nbytes / 2**20:.1f} MiB)")
    return data


def quantize_model(args):
    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as e:
        sys.exit(f"nvidia-modelopt is not available ({e}); install nvidia-modelopt==0.19.0 --no-deps")
    data = prepare_calibration_data(calibration_files(args.calib_list, args.data_root, args.max_images))
    exclude = NODES_TO_EXCLUDE + [r for r in args.exclude_extra.split(",") if r]
    print(f"method {args.calibrate_method}; excluded from INT8: {exclude}")
    t0 = time.time()
    quantize(onnx_path=args.onnx, quantize_mode="int8", calibration_data=data,
             calibration_method=METHOD_MAP[args.calibrate_method], calibration_eps=["cpu"],
             nodes_to_exclude=exclude, output_path=args.output_onnx, use_external_data_format=False, verbose=True)
    size = Path(args.output_onnx).stat().st_size / 2**20
    print(f"quantized ONNX written to {args.output_onnx} ({size:.1f} MiB) in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, help="simplified ONNX model with a static batch size")
    ap.add_argument("--calib_list", required=True, help="calibration list (calibration/calib_<n>.json)")
    ap.add_argument("--data_root", default=str(paths.DATA_ROOT), help="DATA_ROOT of the dataset package")
    ap.add_argument("--output_onnx", default="", help="default: <onnx>_int8.onnx")
    ap.add_argument("--calibrate_method", default="minmax", choices=list(METHOD_MAP))
    ap.add_argument("--max_images", type=int, default=0, help="first N entries of the list (0 = all)")
    ap.add_argument("--exclude_extra", default="", help="extra node regexes kept out of INT8, comma-separated")
    args = ap.parse_args()
    if not args.output_onnx:
        args.output_onnx = str(Path(args.onnx).with_name(Path(args.onnx).stem + "_int8.onnx"))
    quantize_model(args)


if __name__ == "__main__":
    main()
