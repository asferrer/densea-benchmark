"""Tests of variant_pilot: the object crops of copy-paste variants include only the pasted objects (no images)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "synth_quality"))
import variant_pilot as ab  # noqa: E402


def _variant(tmp_path, with_origin):
    d = {"images": [{"id": 0, "file_name": "a.jpg", "width": 100, "height": 80}],
         "categories": [{"id": 0, "name": "Bottle"}, {"id": 1, "name": "Can"}],
         "info": {"synth_generator": {"rect_filter": {"removed_by_class": {"Fishing_Net": 3}}}},
         "annotations": [{"id": 1, "image_id": 0, "category_id": 0, "bbox": [1, 1, 30, 30]},
                         {"id": 2, "image_id": 0, "category_id": 1, "bbox": [40, 40, 30, 30]},
                         {"id": 3, "image_id": 0, "category_id": 1, "bbox": [5, 5, 30, 30], "iscrowd": 1}]}
    if with_origin:
        d["annotations"][0]["origin"] = "real"
        d["annotations"][1]["origin"] = "pasted"
    (tmp_path / "synthetic_dataset.json").write_text(json.dumps(d), encoding="utf8")
    return tmp_path


def test_keep_pasted_excludes_only_real_boxes():
    assert ab.keep_pasted({}) and ab.keep_pasted({"origin": "pasted"}) and not ab.keep_pasted({"origin": "real"})


def test_load_variant_drops_kept_real_boxes(tmp_path):
    imgs, anns = ab.load_variant(_variant(tmp_path, True))
    assert [a["cls"] for a in anns] == ["Can"] and len(imgs) == 1


def test_load_variant_without_origin_keeps_everything_but_crowd(tmp_path):
    _, anns = ab.load_variant(_variant(tmp_path, False))
    assert [a["cls"] for a in anns] == ["Bottle", "Can"]


def test_annotation_counts_and_report_mentions_the_exclusion(tmp_path):
    root = tmp_path / "root"
    _variant((root / "g_v11_copypaste").mkdir(parents=True) or root / "g_v11_copypaste", True)
    counts = ab.annotation_counts(root, ["g_v11_copypaste"])
    assert counts["g_v11_copypaste"] == {"pasted": 1, "kept_real": 1,
                                         "rect_dropped": {"Fishing_Net": 3}, "images": 1,
                                         "visibility_dropped": {}, "visibility_rejections": {}}
    text = "\n".join(ab.note_lines(counts))
    assert "only the pasted objects" in text and "image-level AUC is not comparable" in text and "Fishing_Net 3" in text


def test_q_variants_are_listed_and_use_real_train_backgrounds():
    q = ["q_base", "q_lab", "q_lab_same", "q_lab_local"]
    r = ["r_matte", "r_matte_fg", "r_medium", "r_medium_harm"]
    # q and r follow each other in this order; later variants (s_clean, s_drop) may come after
    i = ab.VARIANTS.index("q_base")
    assert ab.VARIANTS[i:i + 8] == q + r
    assert all(v in ab.REAL_BACKGROUND for v in r)
    assert all(v in ab.REAL_BACKGROUND for v in q)
    assert ab.av.REAL_BG_VARIANTS is ab.REAL_BACKGROUND


def test_refined_variant_folders_count_as_real_background():
    assert "p_pos_tight_r030_c10" in ab.REAL_BACKGROUND and "c_v10_noaug" not in ab.REAL_BACKGROUND
