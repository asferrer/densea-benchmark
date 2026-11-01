"""Evaluation of runs on TEST (eval/evaluate_runs.py): subsets, shared resamples and default pairs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from eval import evaluate_runs as ER  # noqa: E402
from eval.coco_tools import load_gt  # noqa: E402


def write_gt(path):
    sources = ["CleanSea", "CleanSea", "Neural_Ocean", "Ocean_garbage"]
    images = [{"id": i, "file_name": f"images/s/{i}.jpg", "width": 100, "height": 100, "source": s}
              for i, s in enumerate(sources)]
    json.dump({"images": images, "annotations": [], "categories": [{"id": 0, "name": "Bottle"}]}, open(path, "w"))


def test_subsets_split_by_source(tmp_path):
    write_gt(tmp_path / "gt.json")
    sub = ER.subsets(load_gt(tmp_path / "gt.json"))
    assert sub["full"] == [0, 1, 2, 3] and sub["CleanSea"] == [0, 1] and sub["Roboflow"] == [2, 3]


def test_weights_are_shared_and_resample_the_subset():
    allids = np.arange(10)
    w0, w = ER.weights("full", allids[:6], allids)
    w0b, wb = ER.weights("full", allids[:6], allids)
    assert np.array_equal(w, wb) and w0.tolist() == [1] * 6 + [0] * 4
    assert (w.sum(1) == 6).all() and (w[:, 6:] == 0).all()


def test_default_pairs_compare_with_real_only_of_the_same_detector():
    runs = {"RT_exp07_real100": {"detector": "rtdetrv2", "strategy": "exp07_real100"},
            "RT_exp08_real_synth_minor": {"detector": "rtdetrv2", "strategy": "exp08_real_synth_minor"},
            "DD_exp08_real_synth_minor": {"detector": "diffusiondet", "strategy": "exp08_real_synth_minor"}}
    assert ER.default_pairs(runs) == [["RT_exp08_real_synth_minor", "RT_exp07_real100"]]


def test_test_split_needs_confirmation(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ALLOW_TEST"}
    out = subprocess.run([sys.executable, str(KIT / "eval" / "evaluate_runs.py")], capture_output=True, text=True,
                         env=env, cwd=tmp_path)
    assert out.returncode != 0 and "ALLOW_TEST=1" in out.stderr
