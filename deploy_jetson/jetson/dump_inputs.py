"""Real inputs for ``trtexec --loadInputs``: the first VAL image (by id) whose prepared shape is 800 x 800 (2,021 of
the 2,185 VAL images), with its noise (``torch.manual_seed(image id)``). Writes raw float32 files
``<out>/dd_in_{image,images_whwh,noise}.bin``. Random inputs do not measure the latency of DiffusionDet, because the
boxes decide the pyramid level of each RoI.

Run it from the ``dd/`` folder with the DiffusionDet virtualenv (``DD_CODE_DIR`` and ``DD_CONFIG`` set).
Usage: python dump_inputs.py --val data/val.json --data_root data --out models
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

import export_onnx_diffusiondet as ex


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = ex.get_config()
    with open(a.val, encoding="utf8") as f:
        images = sorted(json.load(f)["images"], key=lambda d: d["id"])
    for im in images:
        img = cv2.imread(os.path.join(a.data_root, im["file_name"]))[:, :, ::-1].copy()
        image, whwh, _ = ex.prepare(img, cfg)
        if image.shape[-2:] == (800, 800):
            break
    torch.manual_seed(im["id"])
    noise = torch.randn(1, cfg.MODEL.DiffusionDet.NUM_PROPOSALS, 4).numpy()
    for name, arr in (("image", image), ("images_whwh", whwh), ("noise", noise)):
        np.ascontiguousarray(arr, np.float32).tofile(os.path.join(a.out, f"dd_in_{name}.bin"))
    print("image", im["id"], image.shape)


if __name__ == "__main__":
    main()
