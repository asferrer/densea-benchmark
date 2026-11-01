"""Build the training JSON of phase 7, the incremental series with both batches (see prepare/experiments.py).

Same design as phase 4 (prepare/prepare_proxy.py) at 25 % and 50 % real:
  inc_realNN      only the real fraction (stratified_subset, seed 21: the same images as exp12/exp13)
  inc_v11_realNN  real fraction + copy-paste batch of that fraction (DATA_ROOT/synthetic/copy_paste_rNN/,
                  generated with the real fraction as backgrounds and merged with synthetic/merge_chunks.py)
  inc_v9_realNN   real fraction + random images of the clean-background batch (seed 21) until the pasted
                  objects of the copy-paste batch of the same fraction are matched (field `match`)

The script checks that the copy-paste batch uses only images of its fraction as backgrounds: the
`background` field of every image (`<id:05d>_<name>`) must carry an id of the fraction, otherwise it stops.
Usage: python prepare/prepare_series.py [--experiments inc_real25 ...]
Outputs in PREPARE_OUT: <exp>/{train,val,test}.json and SERIES_SUMMARY.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from prepare import strategies as st  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402
from prepare.prepare_proxy import pasted, sample_to_pasted, summary_row  # noqa: E402
from prepare.prepare_strategies import OUT, load_real, load_synth, write_exp  # noqa: E402


def foreign_backgrounds(lot_ds, sub):
    """Images of the copy-paste batch whose background is not in the real fraction `sub` (must be 0)."""
    ids = {im["id"] for im in sub["images"]}
    return sum(int(im["background"].split("_", 1)[0]) not in ids for im in lot_ds["images"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--rfdetr-root", default="/data")
    ap.add_argument("--seed", type=int, default=paths.SEED)
    args = ap.parse_args()
    real = load_real()
    cats = real["train"]["categories"]
    exps = {k: v for k, v in EXPERIMENTS.items() if v["phase"] == 7
            and (not args.experiments or k in args.experiments)}
    lots = {e["lot"] for e in EXPERIMENTS.values() if e["phase"] == 7 and e["synth"] != "none"}
    lots |= {e["match"] for e in EXPERIMENTS.values() if e["phase"] == 7 and "match" in e}
    synth_lots = {lot: load_synth(lot, cats)[0] for lot in sorted(lots)}
    summary = {"experiments": {}}
    for name, exp in exps.items():
        sub = st.stratified_subset(real["train"], exp["real"], args.seed)
        synth = None
        if exp["synth"] == "none":
            train = {**sub, "images": [dict(im, source="real") for im in sub["images"]]}
        else:
            if exp["lot"] == "v9":
                target = len(pasted(synth_lots[exp["match"]]))
                synth = sample_to_pasted(synth_lots["v9"], target, args.seed)
                summary["experiments"].setdefault(name, {})["pasted_target"] = target
            else:
                synth = synth_lots[exp["lot"]]
                bad = foreign_backgrounds(synth, sub)
                if bad:
                    sys.exit(f"{name}: {bad} images of batch {exp['lot']} use backgrounds outside the real fraction")
            train = st.merge(synth, sub, name, exp["lot"])
        write_exp(name, train, real, args.rfdetr_root)
        row = summary["experiments"].setdefault(name, {})
        row.update(summary_row(train, synth), real_images=len(sub["images"]))
        print(name, row["images"], "images,", row["real_images"], "real,", row["pasted_annotations"], "pasted objects")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "SERIES_SUMMARY.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
