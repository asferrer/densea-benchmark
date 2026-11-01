"""Write the evaluation manifest `eval/runs.json`: one run per (detector, experiment) and the pairs to compare.

The predictions of each run are read by `eval/evaluate_runs.py` from
`common.paths.predictions_dir(detector, "test") / <experiment>_test_preds.json`
(see `eval.coco_tools.preds_file`).
Pairs (A - B, within each detector): each strategy minus Real-Only (exp07); the two best strategies
against the other strategies; in the incremental series, synthetic + X% real minus X% real
(exp02-exp11, exp03-exp12, exp04-exp13, exp05-exp14) and exp15 - exp11.
Usage: python eval/make_manifest.py [--out eval/runs.json]
"""
import argparse
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from eval.coco_tools import DETECTORS  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402

SHORT = {"rtdetrv2": "RT", "diffusiondet": "DD", "rfdetr": "RF"}
INCR = [("exp02_synth100_real10", "exp11_real10"), ("exp03_synth100_real25", "exp12_real25"),
        ("exp04_synth100_real50", "exp13_real50"), ("exp05_synth100_real75", "exp14_real75"),
        ("exp15_real10_synth_invprop", "exp11_real10")]


def detector_pairs(short):
    ref = f"{short}_exp07_real100"
    pairs = [[f"{short}_{e}", ref] for e in ("exp06_synth100_real100", "exp08_real_synth_minor",
                                              "exp09_real_synth_invprop")]
    pairs += [[f"{short}_exp06_synth100_real100", f"{short}_exp08_real_synth_minor"],
              [f"{short}_exp06_synth100_real100", f"{short}_exp09_real_synth_invprop"]]
    return pairs + [[f"{short}_{a}", f"{short}_{b}"] for a, b in INCR]


def build():
    runs, pairs = {}, []
    for det in DETECTORS:
        runs.update({f"{SHORT[det]}_{exp}": {"detector": det, "strategy": exp} for exp in EXPERIMENTS})
        pairs += detector_pairs(SHORT[det])
    return {"runs": runs, "pairs": pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs.json"))
    args = ap.parse_args()
    man = build()
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(man, f, indent=1)
    print("written", args.out, len(man["runs"]), "runs,", len(man["pairs"]), "pairs")


if __name__ == "__main__":
    main()
