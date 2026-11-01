"""Export an RF-DETR (Base) checkpoint to ONNX for TensorRT and check the equivalence. Uses VAL only, never TEST.

The model is rebuilt from the ``args`` stored in the checkpoint, as in the code that produced the reference VAL
predictions: classification head sized from the checkpoint and queries cut to ``num_queries x group_detr``. Then
``export()`` switches the model to its export forward (a single query group) and ``torch.onnx.export`` writes it
(opset 17, static shape 1 x 3 x 560 x 560, constant folding).
  input:   ``input`` (1, 3, 560, 560), normalized image (``eval_rfdetr_val.prepare``);
  outputs: ``dets`` (1, 300, 4) normalized cxcywh and ``labels`` (1, 300, C) logits. The post-processing
           (``eval_rfdetr_val.postprocess``) stays outside the engine.

Checks with ``--check N`` (the first N VAL images by id):
  1) ``prepare`` against the VAL transforms of the library: maximum input difference;
  2) original forward (NestedTensor) against the export forward: maximum logit and box difference;
  3) ONNX (onnxruntime, CPU) against the export forward;
  4) top-100 detections against the reference predictions of the PC (``--ref``): score difference and
     number of equal classes.

Run it on the CPU inside the densea-rfdetr container, with ``onnx`` and ``onnxruntime`` available:
    python /kit/deploy_jetson/export_onnx_rfdetr.py --ckpt /runs/rfdetr/<exp>/checkpoint_best_ema.pth \
        --out /runs/deploy_jetson/models/rfdetr_<exp>.onnx --check 3
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from eval_rfdetr_val import postprocess, prepare  # noqa: E402

RFDETR_SRC = os.environ.get("RFDETR_SRC", "/app/densea-rfdetr/src")   # source folder of the RF-DETR library
sys.path.insert(0, RFDETR_SRC)


def load(ckpt_path):
    """Rebuild the model of a checkpoint on the CPU; returns (model in eval mode, args, epoch)."""
    import torch
    from rfdetr.models import build_model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.pretrain_weights, args.device, args.distributed = None, "cpu", False
    model = build_model(args)
    state = ckpt["model"]
    n_cls = state["class_embed.bias"].shape[0]
    if n_cls != args.num_classes + 1:
        model.reinitialize_detection_head(n_cls)
    nq = args.num_queries * args.group_detr
    for key in list(state):
        if key.endswith("refpoint_embed.weight") or key.endswith("query_feat.weight"):
            state[key] = state[key][:nq]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing keys {len(missing)}, unexpected {len(unexpected)}; epoch {ckpt.get('epoch')}")
    return model.eval(), args, ckpt.get("epoch")


def lib_input(img, res):
    from rfdetr.datasets import transforms as T
    tf = T.Compose([T.SquareResize([res]), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return tf(img, None)[0][None]


def check_image(im, model, exp_model, sess, args, ref):
    import torch
    from PIL import Image
    from rfdetr.util.misc import nested_tensor_from_tensor_list
    img = Image.open(paths.DATA_ROOT / im["file_name"]).convert("RGB")
    x = prepare(img, args.resolution)
    d_in = float(np.abs(x - lib_input(img, args.resolution).numpy()).max())
    with torch.no_grad():
        o = model(nested_tensor_from_tensor_list([torch.from_numpy(x[0])]))
        dets, logits = exp_model(torch.from_numpy(x))
    d_pt = max(float((o["pred_logits"] - logits).abs().max()), float((o["pred_boxes"] - dets).abs().max()))
    od, ol = sess.run(["dets", "labels"], {"input": x})
    d_onnx = max(float(np.abs(ol - logits.numpy()).max()), float(np.abs(od - dets.numpy()).max()))
    s, lab, _ = postprocess(od[0], ol[0], np.array(img.size, dtype=np.float64), args.num_select)
    top = sorted((p for p in ref if p["image_id"] == im["id"]), key=lambda p: -p["score"])[:100]
    d_ref = max(abs(p["score"] - float(si)) for p, si in zip(top, s[:100]))
    same = sum(p["category_id"] == int(li) for p, li in zip(top, lab[:100]))
    print(f"id {im['id']}: input {d_in:.2e} | forward/export {d_pt:.2e} | onnx {d_onnx:.2e} | "
          f"score vs reference {d_ref:.2e}, same class {same}/100", flush=True)


def check(model, exp_model, onnx_path, args, n, ref_path):
    import onnxruntime as ort
    with open(paths.annotations_path("val"), encoding="utf8") as f:
        val = json.load(f)
    with open(ref_path, encoding="utf8") as f:
        ref = json.load(f)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for im in sorted(val["images"], key=lambda d: d["id"])[:n]:
        check_image(im, model, exp_model, sess, args, ref)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--ref", default=str(paths.predictions_dir("rfdetr", "val") / "exp07_real100_val_preds.json"),
                    help="reference VAL predictions of the PC, used by --check")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, args, _ = load(a.ckpt)
    exp_model = copy.deepcopy(model)
    exp_model.export()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    x = torch.zeros(1, 3, args.resolution, args.resolution)
    with torch.no_grad():
        torch.onnx.export(exp_model, x, a.out, input_names=["input"], output_names=["dets", "labels"],
                          export_params=True, keep_initializers_as_inputs=False, do_constant_folding=True,
                          opset_version=a.opset, dynamo=False)
    meta = {"ckpt": a.ckpt, "resolution": args.resolution, "num_select": args.num_select, "opset": a.opset,
            "num_queries": args.num_queries, "encoder": args.encoder}
    with open(a.out.replace(".onnx", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("ONNX:", a.out, meta, flush=True)
    if a.check:
        check(model, exp_model, a.out, args, a.check, a.ref)


if __name__ == "__main__":
    main()
