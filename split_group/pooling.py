"""Pool the public annotation files into one list of image rows.

The three COCO files of DenSea-Aggregated (train, val, test) are merged into a single list ordered by
image id. That order is the row order of the descriptor files (``dhash.npy``, ``embeddings.npy``) and of
every array built from the rows.
"""
import collections
import json
from pathlib import Path

from common import paths


def pool_annotations(annotations_dir=None):
    """Return ``(rows, classes)`` pooled from ``train.json``, ``val.json`` and ``test.json``.

    rows: list of dicts ``{id, file_name, source, file_split, group}`` ordered by image id, where
    ``file_split`` is the file the image comes from and ``group`` is the group id stored in the file
    (``None`` when absent).
    classes: list aligned with ``rows``; each item is the list of category ids of the image's boxes.
    """
    annotations_dir = Path(annotations_dir or paths.ANNOTATIONS_DIR)
    rows, cats = {}, collections.defaultdict(list)
    for split in paths.SPLITS:
        with open(annotations_dir / f"{split}.json", encoding="utf8") as f:
            coco = json.load(f)
        for im in coco["images"]:
            if im["id"] in rows:
                raise ValueError(f"image id {im['id']} appears in more than one file")
            rows[im["id"]] = {"id": im["id"], "file_name": im["file_name"], "source": im["source"],
                              "file_split": split, "group": im.get("group")}
        for a in coco["annotations"]:
            cats[a["image_id"]].append(a["category_id"])
    ordered = [rows[i] for i in sorted(rows)]
    return ordered, [cats[r["id"]] for r in ordered]
