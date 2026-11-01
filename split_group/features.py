"""Per-image content descriptors used to find near-duplicates.

  - dHash 16x16 (17x16 grayscale, LANCZOS, on the thumbnail <= 448 px) of the 8 dihedral transforms;
  - ResNet-50 embedding (torchvision IMAGENET1K_V2, fc = identity) at 224x224, mean over the 8
    transforms, L2-normalised.

Script usage (images are read from ``DATA_ROOT/<file_name>``; about 1 h on CPU, 10 min on GPU):
  python split_group/features.py [--annotations-dir DIR] [--out DIR] [--device cuda]
Writes ``dhash.npy`` and ``embeddings.npy`` (see ``split_group/content_edges.py``), one row per image of
the pooled annotations in ascending image id.
"""
import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from split_group.content_edges import DHASH_FILE, EMBEDDINGS_FILE  # noqa: E402
from split_group.pooling import pool_annotations  # noqa: E402

TR = [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180,
      Image.ROTATE_90, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dhash(gray, n=16):
    a = np.asarray(gray.resize((n + 1, n), Image.LANCZOS), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).flatten()


def read_bytes(p, tries=6):
    for k in range(tries):
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1 + 2 * k)


def prep(p):
    """(dHash 8x256 bool, image 224x224x3 float) of one file."""
    im = Image.open(io.BytesIO(read_bytes(p))).convert("RGB")
    im.thumbnail((448, 448))
    g = im.convert("L")
    hd = np.stack([dhash(g if t is None else g.transpose(t)) for t in TR])
    x = np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    return hd, x


def build_model(device):
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval().to(device)
    m.fc = torch.nn.Identity()
    return m


def embed(m, x, device):
    """Mean embedding of the 8 dihedral transforms, L2-normalised."""
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    xs = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3), x.transpose(2, 3),
                    x.transpose(2, 3).flip(3), x.transpose(2, 3).flip(2), x.transpose(2, 3).flip(2).flip(3)])
    f = m((xs - MEAN.to(device)) / STD.to(device)).mean(0)
    return torch.nn.functional.normalize(f, dim=0).cpu().numpy()


def compute(paths_list, device="cpu", workers=16, log_every=500):
    """Return ``(dHash N x 8 x 256 bool, embeddings N x 2048 float32)``."""
    m = build_model(device)
    hd = np.zeros((len(paths_list), 8, 256), bool)
    em = np.zeros((len(paths_list), 2048), np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, torch.no_grad():
        for i, (h, x) in enumerate(ex.map(prep, paths_list)):
            hd[i], em[i] = h, embed(m, x, device)
            if i % log_every == 0:
                print(i, f"{time.time() - t0:.0f}s", flush=True)
    return hd, em


def save_features(out, hd, em):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / DHASH_FILE, np.packbits(hd, axis=2))
    np.save(out / EMBEDDINGS_FILE, em.astype(np.float16))


def compute_for_annotations(annotations_dir, data_root, device="cpu"):
    """Descriptors of every image of the pooled annotations, read from ``data_root``."""
    rows, _ = pool_annotations(annotations_dir)
    return compute([str(Path(data_root) / r["file_name"]) for r in rows], device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-dir", default=str(paths.ANNOTATIONS_DIR))
    ap.add_argument("--data-root", default=str(paths.DATA_ROOT))
    ap.add_argument("--out", default=str(paths.FEATURES_DIR))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    hd, em = compute_for_annotations(args.annotations_dir, args.data_root, args.device)
    save_features(args.out, hd, em)
    print("done", len(hd))


if __name__ == "__main__":
    main()
