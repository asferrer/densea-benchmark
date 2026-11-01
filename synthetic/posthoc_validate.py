"""Post-hoc quality validation of a synthetic batch against real reference images.

The generator's batch path validates without reference images (a self-consistency heuristic); the endpoint
``POST /validate`` of the augmentor accepts ``reference_images``. This script samples N synthetic images from
the COCO file of a batch and validates each against 5 real images drawn at random (seeded) from the train
split, never from val or test. It only reports: it does not remove images.

Paths are those seen FROM the augmentor container: ``--job-images`` is the folder of the synthetic images and
``--reference-root`` the folder that ``file_name`` of the train annotations is relative to.

Usage (augmentor running):
  python synthetic/posthoc_validate.py --coco <synthetic_dataset.json> --job-images <container images folder> \
      --reference-root <container dataset root> [--n 300] [--out posthoc_validate.csv]
"""
import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def train_files():
    """`file_name` of every image of the train split (relative to the dataset root)."""
    with open(paths.annotations_path("train"), encoding="utf8") as f:
        return [im["file_name"] for im in json.load(f)["images"]]


def annotations_of(coco, img_id, cats):
    return [{"x": int(a["bbox"][0]), "y": int(a["bbox"][1]), "width": int(a["bbox"][2]),
             "height": int(a["bbox"][3]), "class_name": cats[a["category_id"]]}
            for a in coco["annotations"] if a["image_id"] == img_id]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coco", required=True, help="synthetic_dataset.json of the batch")
    ap.add_argument("--job-images", required=True, help="synthetic images folder as seen from the container")
    ap.add_argument("--reference-root", required=True, help="dataset root as seen from the container")
    ap.add_argument("--url", default="http://localhost:8004/validate")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--out", default="posthoc_validate.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    coco = json.load(open(args.coco, encoding="utf8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    train = train_files()
    root = args.reference_root.rstrip("/")
    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "perceptual", "anomaly", "composition", "overall_pass"])
        for im in rng.sample(coco["images"], min(args.n, len(coco["images"]))):
            refs = [f"{root}/{p}" for p in rng.sample(train, 5)]
            res = post(args.url, {"image_path": f"{args.job_images}/{im['file_name']}",
                                  "annotations": annotations_of(coco, im["id"], cats),
                                  "reference_images": refs, "check_physics": False})
            q = res.get("quality_score", res)
            w.writerow([im["file_name"], q.get("perceptual_quality"), q.get("anomaly_score"),
                        q.get("composition_score"), q.get("overall_pass")])
    print("written", args.out)


if __name__ == "__main__":
    main()
