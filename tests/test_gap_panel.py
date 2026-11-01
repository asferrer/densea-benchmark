"""Tests of gap_panel, gap_panel_validate and gap_extractors with toy data (no images, no network, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import gap_extractors as gx  # noqa: E402
import gap_panel as gp  # noqa: E402
import gap_panel_validate as gv  # noqa: E402


def _pack(n_per, shift, seed, prefix):
    rng = np.random.default_rng(seed)
    cls = [c for c in gp.dg.CLASSES for _ in range(n_per)]
    centers = {c: 3 * i for i, c in enumerate(gp.dg.CLASSES)}
    f = np.array([rng.normal(size=6) + np.eye(6)[i % 6] * centers[c] / 3 + shift for i, c in enumerate(cls)])
    meta = [{"cls": c, "img": i, "path": f"{prefix}{i}", "bbox": [0, 0, 20, 20]} for i, c in enumerate(cls)]
    return {"f": f, "meta": meta, "g": [f"{prefix}{i // 2}" for i in range(len(cls))], "cls": cls}


def test_registry_has_the_five_extractors_and_licenses():
    assert set(gx.ORDER) | set(gx.TASK) | set(gx.TASK_SENS) == set(gx.REGISTRY) and len(gx.ORDER) == 5
    assert all(s.license for s in gx.REGISTRY.values()) and gx.REGISTRY["dinov3_b16"].gated
    assert gx.ORDER.index("resnet50") < gx.ORDER.index("dinov2_b14") < gx.ORDER.index("siglip2_b16") < gx.ORDER.index("clip_b16")


def test_items_hash_changes_with_the_box_and_the_order():
    a = [{"path": "x", "bbox": [0, 0, 1, 1]}, {"path": "y", "bbox": [1, 1, 2, 2]}]
    assert gx.items_hash(a) == gx.items_hash(list(a)) and gx.items_hash(a) != gx.items_hash(a[::-1])
    assert gx.items_hash(a) != gx.items_hash([a[0], dict(a[1], bbox=[1, 1, 2, 3])])


def test_cache_at_switches_and_restores_the_cache_folder(tmp_path):
    main = gx.CACHE
    with gx.cache_at(tmp_path):
        assert gx.CACHE == tmp_path
    assert gx.CACHE == main


def test_tensor_normalizes_and_changes_to_nchw():
    from PIL import Image
    t = gx._tensor([Image.new("RGB", (224, 224), (255, 255, 255))] * 2, gx.HALF)
    assert tuple(t.shape) == (2, 3, 224, 224) and abs(float(t.mean()) - 1.0) < 1e-6


def test_panel_variant_distinguishes_shifted_synthetic_from_a_faithful_one():
    real, real_val = _pack(40, 0.0, 1, "r"), _pack(40, 0.0, 2, "v")
    q = {c: 40 for c in gp.dg.CLASSES}
    good = gp.panel_variant(real, real_val, _pack(40, 0.0, 3, "s"), q, real_val, ci=False)
    bad = gp.panel_variant(real, real_val, _pack(40, 2.5, 4, "t"), q, real_val, ci=False)
    assert bad["global"]["auc"] > good["global"]["auc"] + 0.2
    assert bad["global"]["cmmd"] > good["global"]["cmmd"] and bad["global"]["kid"] > good["global"]["kid"]
    assert bad["global"]["coverage@val"] < good["global"]["coverage@val"]
    assert set(good["per_class"]) == set(gp.dg.CLASSES) and "tstr_recall" in good["per_class"]["Bottle"]


def test_batch_of_an_arm_with_seed_suffix():
    assert gv.lot_of("px_v9_real10_s2", gp.LOTS) == "px_v9" and gv.lot_of("px_q_real10", gp.LOTS) == "px_q"
    with pytest.raises(KeyError):
        gv.lot_of("other", gp.LOTS)


def test_agreement_uses_the_sense_of_the_metric():
    ap = {"a": 35.0, "b": 36.6, "c": 36.1}
    r = gv.agreement({"a": 0.9, "b": 0.5, "c": 0.6}, ap, "auc")          # high AUC = worse
    assert r["pairs"] == [3, 3] and r["order"][0] == "b" and r["spearman"] == pytest.approx(1.0)
    r2 = gv.agreement({"a": 0.9, "b": 0.5, "c": 0.6}, ap, "coverage@val")    # same series with the opposite sense
    assert r2["pairs"] == [0, 3]


def test_validate_ext_with_extra_seed_arms_ignores_pairs_of_the_same_batch():
    def row(v):
        return {"global": {m: v for m in gv.GLOBAL_METRICS}, "per_class": {c: {m: v for m in gv.CLASS_METRICS} for c in gv.CLASSES}}
    ext = {"px_v9": row(0.9), "px_p": row(0.5), "px_q": row(0.6)}
    ap = {e: dict({"AP": a}, **{f"AP_{c}": a for c in gv.CLASSES}) for e, a in
          {"px_v9_real10": 35.0, "px_p_real10": 36.6, "px_q_real10": 36.1, "px_v9_real10_s2": 34.5, "px_p_real10_s2": 37.0}.items()}
    exps = list(ap)
    out = gv.validate_ext(ext, ap, gp.LOTS, exps)
    assert out["global"]["auc"]["pairs"][1] == 8       # 10 pairs minus the 2 of the same batch (v9-v9, p-p)
    assert out["global"]["auc"]["pairs"][0] == 8


def test_ap_table_reads_the_table_and_reports_missing_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(gv, "AP_CACHE", tmp_path / "ap.json")
    monkeypatch.setattr(gv, "preds_file", lambda exp: tmp_path / f"{exp}_val_preds.json")
    table = tmp_path / "ap_table.json"
    table.write_text('{"runs": {"px_v9_real10": {"AP": 35.0}}}', encoding="utf8")
    ap, notes = gv.ap_table(["px_v9_real10", "px_v9_real10_s2"], gp.LOTS, table)
    assert ap == {"px_v9_real10": {"AP": 35.0}} and len(notes) == 1 and "px_v9_real10_s2" in notes[0]


def test_s_drop_rule_verdict():
    import gap_panel_drop as gd

    def g(auc, ci_hi, vendi=20.0, cov=0.5):
        return {"auc": auc, "auc_ci": [auc - 0.03, ci_hi], "vendi": vendi, "coverage@val": cov}
    ok = {f"e{i}": {"control": g(0.85, 0.88), "test": g(0.86, 0.88)} for i in range(5)}
    assert gd.verdict(ok)["adopt"] and gd.verdict(ok)["extractors_not_worse"] == 5
    bad3 = dict(ok)
    for i in range(3):                                     # worse by +0.03 in 3 extractors -> only 2 not worse
        bad3[f"e{i}"] = {"control": g(0.85, 0.88), "test": g(0.88, 0.89)}
    assert not gd.verdict(bad3)["adopt"]
    outside_ci = {"e0": {"control": g(0.85, 0.855), "test": g(0.865, 0.9)}}   # +0.015 (<0.02) but above the control CI
    assert gd.verdict(outside_ci)["per_extractor"]["e0"]["worse"] is True
    div = {f"e{i}": {"control": g(0.85, 0.88), "test": g(0.85, 0.88, vendi=15.0)} for i in range(5)}
    assert not gd.verdict(div)["adopt"] and gd.verdict(div)["extractors_not_less_diverse"] == 0
