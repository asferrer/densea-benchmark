"""Convert the official Swin weights (.pth of microsoft/Swin-Transformer) to the .pkl that DiffusionDet loads.

SwinTransformer.init_weights of DiffusionDet does not load pretrained weights, they arrive through MODEL.WEIGHTS. They
are saved like the official swin_base_patch4_window7_224_22k.pkl: the same keys as the .pth and
matching_heuristics = True, so that detectron2 matches them by suffix with backbone.bottom_up.*.
Inside the densea-diffusiondet container:
  docker exec densea_diffusiondet python /kit/run/convert_swin_d2.py
  docker exec densea_diffusiondet python /kit/run/convert_swin_d2.py --check /kit/configs/diffusiondet/<cfg>.yaml
--check builds the model on the CPU, loads MODEL.WEIGHTS and fails if any backbone key is not loaded.
"""
import argparse
import os
import pickle
import re
import sys

import torch

REPO = "/app/densea-diffusiondet"
MODELS = f"{REPO}/models"
SOURCES = ("swin_small_patch4_window7_224", "swin_tiny_patch4_window7_224")


def convert(name):
    dst = f"{MODELS}/{name}_d2.pkl"
    if os.path.exists(dst):
        print(f"{dst} already exists")
        return
    ckpt = torch.load(f"{MODELS}/{name}.pth", map_location="cpu")
    state = {k: v.numpy() for k, v in ckpt.get("model", ckpt).items()}
    with open(dst, "wb") as f:
        pickle.dump({"model": state, "__author__": "microsoft/Swin-Transformer", "matching_heuristics": True}, f)
    print(f"{dst}: {len(state)} tensors")


def check(cfg_file):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    from train_net_densea import add_densea_configs

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_densea_configs(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.DEVICE = "cpu"
    model = build_model(cfg)
    before = {k: v.clone() for k, v in model.backbone.bottom_up.state_dict().items()}
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    after = model.backbone.bottom_up.state_dict()
    same = [k for k, v in after.items() if v.dtype.is_floating_point and v.numel() > 1 and torch.equal(v, before[k])]
    # norm0-3: per-stage output LayerNorms of the detection version; they do not exist in the classification weights
    new = [k for k in same if re.match(r"norm\d\.", k)]
    same = [k for k in same if k not in new]
    print(f"{cfg_file}: {len(after)} backbone tensors; expected new (norm0-3): {len(new)}; not loaded: {len(same)}")
    for k in same[:20]:
        print("  not loaded:", k)
    sys.exit(1 if same else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", help="YAML config to verify")
    a = ap.parse_args()
    if a.check:
        check(a.check)
    for name in SOURCES:
        convert(name)


if __name__ == "__main__":
    main()
