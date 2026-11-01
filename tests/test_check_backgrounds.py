"""Tests of the background check on toy descriptors."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic import check_backgrounds as cb  # noqa: E402


def _corpus(n=6, dim=64):
    rng = np.random.default_rng(3)
    hd = np.unpackbits(rng.integers(0, 256, (n, 8, 32), dtype=np.uint8), axis=2)
    em = rng.normal(size=(n, dim)).astype(np.float32)
    rows = [{"file_name": f"images/Ocean_garbage/train/{i}.jpg", "source": "Ocean_garbage",
             "file_split": ("train", "val", "test")[i % 3]} for i in range(n)]
    return hd, em / np.linalg.norm(em, axis=1, keepdims=True), rows


def test_background_with_relative_in_val_is_excluded(tmp_path):
    hd, em, rows = _corpus()
    bg_hd = np.stack([hd[1], np.unpackbits(np.random.default_rng(9).integers(0, 256, (8, 32), dtype=np.uint8), axis=1)])
    bg_em = np.stack([em[1], np.random.default_rng(10).normal(size=64).astype(np.float32)])
    bg_em[1] /= np.linalg.norm(bg_em[1])
    d, c = cb.match(bg_hd, bg_em, hd, em, np.zeros(len(rows), bool))
    res = cb.summarize(["near.png", "far.png"], d, c, rows)
    assert res[0]["splits_metric"] == ["val"] and res[1]["splits_metric"] == []
    cb.write_outputs(res, tmp_path, "backgrounds")
    assert (tmp_path / "allowed_backgrounds.txt").read_text().split() == ["far.png"]
    assert "with a relative in val or test: **1**" in (tmp_path / "BACKGROUNDS.md").read_text(encoding="utf8")
