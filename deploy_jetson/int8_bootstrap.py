"""Paired comparison of the TensorRT / ONNX / PyTorch variants of each detector on VAL, with 95% CIs from a GROUP
bootstrap (the same resamples as ``analysis/proxy/px_val_bootstrap.py``). VAL only, never TEST.

Predictions (score > 0.001 for RT-DETRv2) are read from ``<preds>/<name>_val_preds.json``, where ``<preds>`` defaults
to ``RUNS_DIR/deploy_jetson/boot/<tag>`` (``deploy_layout.boot_dir``):
  RT-DETRv2:    trt_fp16, trt_int8mix_stem (``eval_trt_val.py`` on the Jetson), onnx_fp32 (``eval_onnx_val.py``),
                rt_torch (PyTorch FP32 on the CPU of the PC);
  RF-DETR:      rf_torch (PyTorch FP32 on the CPU of the PC), rf_trt_fp32, rf_trt_fp16, rf_trt_qF / qG / qH
                (``eval_rfdetr_val.py``);
  DiffusionDet: dd_torch (wrapper on the Jetson GPU), dd_trt_fp32, dd_trt_mixhead, dd_trt_fp16 (``eval_dd_val.py``).
Usage:
    python deploy_jetson/int8_bootstrap.py --set rtdetrv2_final --tag rt_final_15W
    python deploy_jetson/int8_bootstrap.py --set rfdetr --tag rf_15W
    python deploy_jetson/int8_bootstrap.py --set diffusiondet_final --tag dd_final_15W
Outputs: deploy_jetson/out/int8_bootstrap_<tag>.md and .json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "analysis" / "proxy"))
sys.path.insert(0, str(KIT / "analysis" / "full_val"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent / "out"

_DD_RUNS = {"dd_torch": "PyTorch FP32", "dd_trt_fp32": "TensorRT FP32",
            "dd_trt_mixhead": "TensorRT mixed (FP32 heads)", "dd_trt_fp16": "TensorRT FP16"}
_DD_PAIRS = [("dd_trt_mixhead", "dd_torch"), ("dd_trt_fp32", "dd_torch"), ("dd_trt_fp16", "dd_torch"),
             ("dd_trt_mixhead", "dd_trt_fp32")]

# set name -> (title, {run: label}, [(A, B) pairs compared as A - B])
SETS = {
    "rtdetrv2": ("RT-DETRv2 exp07 (seed 21)",
                 {"onnx_fp32": "ONNX FP32 (PC)", "trt_fp16": "TensorRT FP16", "trt_int8mix_stem": "TensorRT INT8 mixed"},
                 [("trt_int8mix_stem", "trt_fp16"), ("trt_int8mix_stem", "onnx_fp32"), ("trt_fp16", "onnx_fp32")]),
    "rtdetrv2_final": ("RT-DETRv2 exp08_real_synth_minor_s3 (MF clean-bg, seed 23)",
                       {"rt_torch": "PyTorch FP32 (PC)", "trt_fp16": "TensorRT FP16",
                        "trt_int8mix_stem": "TensorRT INT8 mixed"},
                       [("trt_fp16", "rt_torch"), ("trt_int8mix_stem", "trt_fp16"), ("trt_int8mix_stem", "rt_torch")]),
    "diffusiondet": ("DiffusionDet exp07 (seed 21)", _DD_RUNS, _DD_PAIRS),
    "diffusiondet_final": ("DiffusionDet exp06v11_synth100_real100_s3 (Std copy-paste, seed 23)", _DD_RUNS, _DD_PAIRS),
    "rfdetr": ("RF-DETR exp07 (seed 21)",
               {"rf_torch": "PyTorch FP32 (PC)", "rf_trt_fp32": "TensorRT FP32", "rf_trt_fp16": "TensorRT FP16",
                "rf_trt_qF": "TensorRT INT8 F (attention)", "rf_trt_qG": "TensorRT INT8 G (attention + fc1)",
                "rf_trt_qH": "TensorRT INT8 H (attention + MLP)"},
               [("rf_trt_fp16", "rf_torch"), ("rf_trt_fp32", "rf_torch"), ("rf_trt_fp16", "rf_trt_fp32"),
                ("rf_trt_qF", "rf_trt_fp16"), ("rf_trt_qG", "rf_trt_fp16"), ("rf_trt_qH", "rf_trt_fp16")]),
}


def available_runs(runs, preds):
    return [e for e in runs if (Path(preds) / f"{e}_val_preds.json").exists()]


def run_table(runs, labels, res):
    """Markdown table of AP, AP50, AP75 and the bootstrap CI of the AP, and the matching JSON record."""
    md = ["| Model | AP | AP50 | AP75 | 95% CI AP |", "|---|---|---|---|---|"]
    record = {}
    for e in runs:
        o, (lo, hi) = res[e]["official"], 100 * np.percentile(res[e]["APb"], [2.5, 97.5])
        record[e] = {k: round(100 * v, 2) for k, v in o.items() if not isinstance(v, dict)}
        md.append(f"| {labels[e]} | " + " | ".join(f"{100 * o[k]:.2f}" for k in ("AP", "AP50", "AP75"))
                  + f" | [{lo:.2f}; {hi:.2f}] |")
    return md, record


def main():
    from eval.coco_tools import load_gt
    from common import paths
    from full_val_bootstrap import diff_row
    from deploy_layout import boot_dir
    from px_val_bootstrap import METRICS, evaluate, group_weights
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=None, help="folder with <name>_val_preds.json (default: boot/<tag>)")
    ap.add_argument("--tag", required=True, help="measurement set, for example 15W; names the outputs")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--set", default="rtdetrv2", choices=list(SETS))
    a = ap.parse_args()
    preds = a.preds or boot_dir(a.tag)
    title, labels, pairs = SETS[a.set]
    gt = load_gt(paths.annotations_path("val"))
    w, n_groups = group_weights(sorted(gt.getImgIds()), a.b)
    runs = available_runs(labels, preds)
    res = {e: evaluate(e, gt, preds, w) for e in runs}
    out = {"tag": a.tag, "n_groups": n_groups, "B": a.b}
    table, out["runs"] = run_table(runs, labels, res)
    md = [f"# {title} on the Jetson, mode {a.tag}: VAL AP ({n_groups} groups of the group partition; "
          f"group bootstrap, B = {a.b})", ""] + table
    md += ["", "## Paired differences (same group resamples)", "",
           "| A − B | Metric | Difference | 95% CI | p (two-sided) |", "|---|---|---|---|---|"]
    for x, y in [q for q in pairs if q[0] in res and q[1] in res]:
        for m, _ in METRICS:
            md.append(diff_row(f"{labels[x]} − {labels[y]}", 100 * (res[x][m + "0"] - res[y][m + "0"]),
                               100 * (res[x][m + "b"] - res[y][m + "b"]), m, out, "pairs"))
    OUT.mkdir(exist_ok=True)
    (OUT / f"int8_bootstrap_{a.tag}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    with open(OUT / f"int8_bootstrap_{a.tag}.json", "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
