"""Export an RT-DETRv2 checkpoint to ONNX for TensorRT (post-processor included, static batch size 1).

Run it on the CPU inside the RT-DETRv2 container, with ``/app/RT-DETR`` as working directory and ``PYTHONPATH``
(it imports the detector's ``src`` package). The EMA weights of the checkpoint are loaded.

Steps: re-parameterize the RepVGG blocks (``model.deploy()``), replace every ``FrozenBatchNorm2d`` by an eval-mode
``BatchNorm2d``, optionally fuse the Conv+BN pairs (``--fuse-layers``), trace with the dummy inputs
``images (1, 3, 640, 640)`` and ``orig_target_sizes (1, 2) = [width, height]`` (int32), export with
``torch.onnx.export`` (opset 17), check the model (``--check``) and simplify it (``--simplify``, written to
``<output>_simplified.onnx``). The outputs have no batch dimension: ``labels [300]``, ``boxes [300, 4]``,
``scores [300]``.

Example:
    python /kit/deploy_jetson/export_onnx_rtdetrv2.py -c /runs/configs/rtdetrv2/<exp>.yml \
        -r /runs/rtdetrv2/<exp>/best.pth -o /runs/deploy_jetson/models/<name>.onnx \
        --static-batch --fuse-layers --simplify --check
"""
import argparse
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_NAMES = ["images", "orig_target_sizes"]
OUTPUT_NAMES = ["labels", "boxes", "scores"]


def conv_bn_blocks(prefix):
    """Module paths of a CSP block of the encoder: conv1, conv2, three bottlenecks (conv1, conv2 each), conv3."""
    paths = [f"{prefix}.conv1", f"{prefix}.conv2"]
    paths += [f"{prefix}.bottlenecks.{k}.{c}" for k in range(3) for c in ("conv1", "conv2")]
    return paths + [f"{prefix}.conv3"]


def fusion_candidates():
    """Ordered (conv, norm) module-path pairs that may be fused; those missing in the model are skipped later."""
    mods = [f"backbone.conv1.conv1_{k}" for k in (1, 2, 3)]
    for layer in range(4):
        for block in range(2):
            base = f"backbone.res_layers.{layer}.blocks.{block}"
            if block == 0:
                mods.append(f"{base}.short" if layer == 0 else f"{base}.short.conv")
            mods += [f"{base}.branch2a", f"{base}.branch2b"]
    mods += [f"decoder.input_proj.{k}" for k in range(3)] + [f"encoder.input_proj.{k}" for k in range(3)]
    mods += [f"encoder.lateral_convs.{k}" for k in range(2)]
    for k in range(2):
        mods += conv_bn_blocks(f"encoder.fpn_blocks.{k}")
    mods += [f"encoder.downsample_convs.{k}" for k in range(2)]
    for k in range(2):
        mods += conv_bn_blocks(f"encoder.pan_blocks.{k}")
    return [[f"{m}.conv", f"{m}.norm"] for m in mods]


def build_deploy_model(cfg, fuse_layers):
    """Module that maps (images, orig_target_sizes) to the post-processed detections."""
    import torch.nn as nn
    from src.nn.backbone.common import FrozenBatchNorm2d

    def replace_frozen_bn(model):
        for name, module in model.named_children():
            if isinstance(module, FrozenBatchNorm2d):
                bn = nn.BatchNorm2d(module.num_features)
                bn.load_state_dict(module.state_dict())
                setattr(model, name, bn.eval())
            else:
                replace_frozen_bn(module)
        return model

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval()
            self.postprocessor = cfg.postprocessor.eval()
            if hasattr(self.model, "deploy"):
                self.model.deploy()   # fuse the parallel branches of the RepVGG blocks
            self.model = replace_frozen_bn(self.model)
            if fuse_layers:
                self.model = fuse_conv_bn(self.model)

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return DeployModel().eval()


def fuse_conv_bn(model):
    """Fuse the Conv2d + BatchNorm2d pairs of ``fusion_candidates`` that exist in the model."""
    import torch.nn as nn
    from torch.ao.quantization import fuse_modules
    pairs = []
    for conv_path, bn_path in fusion_candidates():
        try:
            conv, bn = model.get_submodule(conv_path), model.get_submodule(bn_path)
        except AttributeError:   # e.g. identity shortcuts have no conv/norm
            continue
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            pairs.append([conv_path, bn_path])
    logging.info("Fusing %d Conv+BN pairs", len(pairs))
    return fuse_modules(model, pairs, inplace=True).eval()


def load_weights(cfg, checkpoint_path):
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("ema") is not None:
        state = ckpt["ema"]["module"] if "module" in ckpt["ema"] else ckpt["ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        sys.exit("the checkpoint has neither 'ema' nor 'model'")
    cfg.model.load_state_dict(state)


def export_onnx(deploy_model, args):
    import torch
    images = torch.randn((1, 3, args.input_height, args.input_width), dtype=torch.float32)
    sizes = torch.tensor([[args.input_width, args.input_height]], dtype=torch.int32)   # [width, height]
    dynamic_axes = None if args.static_batch else {name: {0: "N"} for name in INPUT_NAMES}
    torch.onnx.export(deploy_model, (images, sizes), args.output_file, input_names=INPUT_NAMES,
                      output_names=OUTPUT_NAMES, dynamic_axes=dynamic_axes, opset_version=args.opset,
                      do_constant_folding=True, export_params=True)
    return {INPUT_NAMES[0]: list(images.shape), INPUT_NAMES[1]: list(sizes.shape)}


def simplify_onnx(path, input_shapes):
    import onnx
    import onnxsim
    out = path.replace(".onnx", "_simplified.onnx")
    if out == path:
        out = path.replace(".onnx", ".simpl.onnx")
    model, ok = onnxsim.simplify(onnx.load(path), test_input_shapes=input_shapes)
    if ok:
        onnx.save(model, out)
        logging.info("Simplified model saved to %s", out)
    else:
        logging.warning("Simplification did not validate; no simplified model written")


def main(args):
    from src.core import YAMLConfig
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        load_weights(cfg, args.resume)
    else:
        logging.warning("No checkpoint given (--resume): exporting the initial weights")
    deploy_model = build_deploy_model(cfg, args.fuse_layers)
    shapes = export_onnx(deploy_model, args)
    logging.info("ONNX written to %s", args.output_file)
    if args.check:
        import onnx
        onnx.checker.check_model(onnx.load(args.output_file))
        logging.info("ONNX check passed")
    if args.simplify:
        simplify_onnx(args.output_file, shapes)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", "-c", required=True, help="model YAML configuration")
    p.add_argument("--resume", "-r", help="checkpoint (.pth) to load")
    p.add_argument("--output_file", "-o", default="model.onnx", help="output ONNX file")
    p.add_argument("--check", action="store_true", help="run onnx.checker on the exported model")
    p.add_argument("--simplify", action="store_true", help="simplify with onnx-simplifier")
    p.add_argument("--input-height", type=int, default=640)
    p.add_argument("--input-width", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fuse-layers", action="store_true", help="fuse Conv+BN pairs before exporting")
    p.add_argument("--static-batch", action="store_true", help="static batch size (required for TensorRT on the Jetson)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
