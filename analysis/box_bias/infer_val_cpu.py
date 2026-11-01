"""CPU inference of the best checkpoint (`best.pth`, EMA weights) of several RT-DETRv2 experiments on VAL or TEST of
the group partition. TEST needs --split test and ALLOW_TEST=1: TEST is evaluated once, with the frozen configuration.

It runs INSIDE the densea_rtdetrv2 container (mounts: /kit = this repository, read-only; /runs = RUNS_DIR; /data =
DATA_ROOT), on the CPU and at low priority (run/predict.py launches it):
  docker exec -e CUDA_VISIBLE_DEVICES= densea_rtdetrv2 nice -n 19 python /kit/analysis/box_bias/infer_val_cpu.py \
      --exps exp07_real100 exp06_synth100_real100 [--split val] [--n 0] [--threads 8]
Each image is decoded once and goes through all the models. Preprocessing identical to the VAL preprocessing of
training (ToImage, bilinear Resize 640x640 with antialias, /255). Output per exp:
<out>/<exp>_<split>_preds.json (COCO results format, score > 0.001, as tools/eval.py) and <out>/<split>_ids.json
with the ids of the images processed. Default <out>: /runs/predictions/rtdetrv2/<split>.
"""
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "/app/RT-DETR")
Image.MAX_IMAGE_PIXELS = None


def load(exp):
    from src.core import YAMLConfig  # RT-DETR repository inside the container
    cfg = YAMLConfig(f"/runs/configs/rtdetrv2/{exp}.yml", PResNet={"pretrained": False})
    ck = torch.load(f"/runs/rtdetrv2/{exp}/best.pth", map_location="cpu", weights_only=False)
    model = cfg.model
    model.load_state_dict(ck["ema"]["module"])
    return model.eval(), cfg.postprocessor.deploy().eval(), ck.get("last_epoch")


def prep(path):
    tf = T.Compose([T.ToImage(), T.Resize((640, 640)), T.ToDtype(torch.float32, scale=True)])
    im = Image.open(path).convert("RGB")
    return tf(im), im.size


def pick(images, n):
    """Deterministic sample: sorted by id with a regular step (identical for all exps)."""
    images = sorted(images, key=lambda d: d["id"])
    if n and n < len(images):
        step = len(images) / n
        images = [images[int(i * step)] for i in range(n)]
    return images


def detections(model, post, x, size, image_id):
    """COCO results of one image: detections with score > 0.001."""
    lab, box, sc = post(model(x[None]), size)
    keep = sc[0] > 0.001
    return [{"image_id": image_id, "category_id": int(l), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
            for l, b, s in zip(lab[0][keep].tolist(), box[0][keep].tolist(), sc[0][keep].tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default=None, help="output folder (default /runs/predictions/rtdetrv2/<split>)")
    ap.add_argument("--n", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--datadir", default="/data", help="root of the images (DATA_ROOT)")
    ap.add_argument("--annotations", default=None, help="COCO file of the split (default <datadir>/annotations/<split>.json)")
    a = ap.parse_args()
    if a.split == "test" and os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    out = a.out or f"/runs/predictions/rtdetrv2/{a.split}"
    torch.set_num_threads(a.threads)
    os.makedirs(out, exist_ok=True)
    probe = f"{out}/.write_test"
    open(probe, "w").close()  # fails right away if the destination is read-only
    os.remove(probe)
    ann = json.load(open(a.annotations or f"{a.datadir}/annotations/{a.split}.json", encoding="utf8"))
    imgs = pick(ann["images"], a.n)
    models = {e: load(e) for e in a.exps}
    for e, m in models.items():
        print(e, "last_epoch of the checkpoint:", m[2], flush=True)
    preds = {e: [] for e in a.exps}
    t0 = time.time()
    for k, im in enumerate(imgs):
        x, (w, h) = prep(os.path.join(a.datadir, im["file_name"]))
        with torch.no_grad():
            for e, (model, post, _) in models.items():
                preds[e] += detections(model, post, x, torch.tensor([[w, h]]), im["id"])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(imgs)} {(time.time() - t0) / (k + 1):.2f} s/img", flush=True)
    for e in a.exps:
        json.dump(preds[e], open(f"{out}/{e}_{a.split}_preds.json", "w"))
    json.dump([im["id"] for im in imgs], open(f"{out}/{a.split}_ids.json", "w"))
    print("done", len(imgs), "images", time.time() - t0, "s", flush=True)


if __name__ == "__main__":
    main()
