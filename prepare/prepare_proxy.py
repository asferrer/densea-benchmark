"""Build the training JSON of phase 4, the transfer check of the generator (see prepare/experiments.py).

All arms use the same 10 % real subset as exp11_real10 (stratified_subset, seed 21); VAL and TEST are the
public val.json and test.json (TEST is not evaluated in this phase).
  px_real10      only the 10 % real subset
  px_<variant>   10 % real + a 2,000-image copy-paste batch of the transfer check
                 (DATA_ROOT/synthetic/proxy_<variant>/), pasted onto that same 10 % real subset; every
                 image has pasted objects (`origin` "pasted") and the real objects of the background ("real")
  px_v9_real10   10 % real + random images of the clean-background batch (seed 21) until the mean number of
                 pasted objects of the copy-paste batches is matched (the clean-background batch has only
                 pasted objects), so that arms are compared with the same number of pasted objects

Usage: python prepare/prepare_proxy.py [--experiments px_real10 ...]
Outputs in PREPARE_OUT: <exp>/{train,val,test}.json and PROXY_SUMMARY.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from prepare import strategies as st  # noqa: E402
from prepare.experiments import EXPERIMENTS  # noqa: E402
from prepare.prepare_strategies import OUT, load_real, load_synth, write_exp  # noqa: E402


def pasted(ds):
    """Pasted annotations: those with `origin` "pasted" in copy-paste batches, all in the clean-background batch."""
    return [a for a in ds["annotations"] if a.get("origin", "pasted") == "pasted"]


def sample_to_pasted(ds, target, seed):
    """Random images until `target` pasted objects are reached (the last image may overshoot)."""
    per_img = {}
    for a in pasted(ds):
        per_img[a["image_id"]] = per_img.get(a["image_id"], 0) + 1
    ids = sorted(im["id"] for im in ds["images"])
    random.Random(seed).shuffle(ids)
    keep, n = [], 0
    for i in ids:
        if n >= target:
            break
        keep.append(i)
        n += per_img.get(i, 0)
    return st.filter_images(ds, keep)


def summary_row(train, synth):
    return {"images": len(train["images"]), "annotations": len(train["annotations"]),
            "synthetic_images": len(synth["images"]) if synth else 0,
            "pasted_annotations": len(pasted(synth)) if synth else 0,
            "per_class": dict(st.class_counts(train)),
            "per_class_pasted": dict(st.class_counts({**synth, "annotations": pasted(synth)})) if synth else {}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--rfdetr-root", default="/data")
    ap.add_argument("--seed", type=int, default=paths.SEED)
    args = ap.parse_args()
    real = load_real()
    r10 = st.stratified_subset(real["train"], 0.10, args.seed)
    exps = {k: v for k, v in EXPERIMENTS.items() if v["phase"] == 4
            and (not args.experiments or k in args.experiments)}
    cats = real["train"]["categories"]
    v11 = {e["lot"]: load_synth(e["lot"], cats)[0]
           for e in EXPERIMENTS.values() if e["phase"] == 4 and e.get("lot", "v9") != "v9"}
    target = round(sum(len(pasted(d)) for d in v11.values()) / max(len(v11), 1))
    summary = {"real10_images": len(r10["images"]), "v9_pasted_target": target, "experiments": {}}
    for name, exp in exps.items():
        synth = None
        if exp["synth"] == "none":
            train = {**r10, "images": [dict(im, source="real") for im in r10["images"]]}
        else:
            synth = v11[exp["lot"]] if exp["lot"] != "v9" else \
                sample_to_pasted(load_synth("v9", cats)[0], target, args.seed)
            train = st.merge(synth, r10, name)
        write_exp(name, train, real, args.rfdetr_root)
        summary["experiments"][name] = summary_row(train, synth)
        print(name, summary["experiments"][name]["images"], "images,",
              summary["experiments"][name]["pasted_annotations"], "pasted objects")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "PROXY_SUMMARY.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
