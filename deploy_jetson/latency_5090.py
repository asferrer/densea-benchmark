"""Inference latency on the RTX 5090 (PyTorch, batch 1) of the three models deployed on the Jetson. VAL only.

Run it INSIDE each detector's container (the detector code lives only there), once per model:
    python /kit/deploy_jetson/latency_5090.py --model rtdetrv2          # RT-DETRv2 container
    python /kit/deploy_jetson/latency_5090.py --model rfdetr            # RF-DETR container (cwd /app/densea-rfdetr)
    python /kit/deploy_jetson/latency_5090.py --model diffusiondet      # DiffusionDet container
Each run writes ``RUNS_DIR/deploy_jetson/latency_5090/<model>.json``. Then, on the host,
``python deploy_jetson/latency_5090.py --report`` merges the JSON files into ``out/latency_5090.md`` and ``.json``.
Run it with ``run/gpu_guard/gpu_monitor.py`` active.

Per model and configuration (FP32 / AMP fp16 autocast, each under inference_mode and no_grad):
  - network: tensor already preprocessed on the GPU, 50 warm-up iterations and N timed ones with CUDA events;
  - end to end (e2e): decoded image in memory (PIL / numpy) -> CPU preprocessing -> H2D -> network ->
    post-processing -> result on the CPU, timed with perf_counter and synchronization (JPEG decoding excluded);
  - power: sustained loop of the network (batch 1, synchronizing every iteration) for ``--power-s`` seconds with
    nvidia-smi sampled every 100 ms, discarding the first 4 s. Energy per image = mean power x time per iteration.
    The idle power (model loaded, no inference) gives the net energy.
Inputs: square VAL images (the 800 x 800 shape of DiffusionDet after ResizeShortestEdge and padding, which is the
shape of 2,021 of the 2,185 VAL images), the same ones (by id) for the three models.
Models: RT-DETRv2 ``exp08_real_synth_minor_s3``, RF-DETR ``exp07_real100``, DiffusionDet
``exp06v11_synth100_real100_s3`` (``dd/diffusiondet_exp06v11s3_ema.pth``: the ``ema_state`` of ``model_best.pth``).
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402
from deploy_layout import DD_DIR, LATENCY_DIR  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
RT_EXP = "exp08_real_synth_minor_s3"
RF_CKPT = paths.RUNS_DIR / "rfdetr" / "exp07_real100" / "checkpoint_best_ema.pth"
DD_WEIGHTS = DD_DIR / "diffusiondet_exp06v11s3_ema.pth"
RFDETR_SRC = os.environ.get("RFDETR_SRC", "/app/densea-rfdetr/src")
RTDETR_DIR = os.environ.get("RTDETR_DIR", "/app/RT-DETR")


def square_images(n):
    with open(paths.annotations_path("val"), encoding="utf8") as f:
        val = json.load(f)
    ims = [i for i in sorted(val["images"], key=lambda d: d["id"]) if i["width"] == i["height"]][:n]
    return [(i["id"], Image.open(paths.DATA_ROOT / i["file_name"]).convert("RGB")) for i in ims]


# ---------------------------------------------------------------- adapters: each gives net(x), prep(img), post(out)
def make_rtdetrv2(imgs):
    import torchvision.transforms.v2 as T
    sys.path.insert(0, RTDETR_DIR)
    from src.core import YAMLConfig
    cfg = YAMLConfig(str(paths.RUNS_DIR / "configs" / "rtdetrv2" / f"{RT_EXP}.yml"), PResNet={"pretrained": False})
    ck = torch.load(paths.RUNS_DIR / "rtdetrv2" / RT_EXP / "best.pth", map_location="cpu", weights_only=False)
    model = cfg.model
    model.load_state_dict(ck["ema"]["module"])
    model, post = model.cuda().eval(), cfg.postprocessor.deploy().cuda().eval()
    tf = T.Compose([T.ToImage(), T.Resize((640, 640)), T.ToDtype(torch.float32, scale=True)])

    def prep(img):
        return tf(img)[None].cuda(non_blocking=False), torch.tensor([[img.size[0], img.size[1]]]).cuda()

    def postp(out, sz):
        lab, box, sc = post(out, sz)
        return lab.cpu(), box.cpu(), sc.cpu()
    return dict(net=model, prep=prep, post=postp, in_shape="1x3x640x640", extra=f"epoch {ck.get('last_epoch')}")


def make_rfdetr(imgs):
    sys.path.insert(0, RFDETR_SRC)
    from eval_rfdetr_val import prepare
    from export_onnx_rfdetr import load as load_rfdetr
    from rfdetr.util.misc import nested_tensor_from_tensor_list
    model, args, epoch = load_rfdetr(RF_CKPT)
    model = model.cuda().eval()
    res = args.resolution

    class Net(torch.nn.Module):
        def forward(self, x):
            o = model(nested_tensor_from_tensor_list([x[0]]))
            return o["pred_logits"], o["pred_boxes"]

    def prep(img):
        return torch.from_numpy(prepare(img, res)).cuda(), torch.tensor([img.size[0], img.size[1]] * 2).cuda()

    def postp(out, wh):   # library PostProcess: sigmoid, top-300 over queries x classes, cxcywh -> xyxy
        logits, boxes = out
        prob = logits.float().sigmoid().flatten(1)
        sc, idx = prob.topk(300, dim=1)
        q, lab = idx // logits.shape[2], idx % logits.shape[2]
        b = boxes.float()[0][q[0]]
        xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1) * wh
        return sc.cpu(), lab.cpu(), xyxy.cpu()
    return dict(net=Net().eval(), prep=prep, post=postp, in_shape=f"1x3x{res}x{res}", extra=f"epoch {epoch}")


def make_diffusiondet(imgs):
    import export_onnx_diffusiondet as ex
    cfg, model = ex.load(str(DD_WEIGHTS), device="cuda")
    core = ex.DDCore(model).cuda().eval()
    n_prop = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
    state = {"hw": None}

    class Net(torch.nn.Module):   # inputs (image, whwh, noise) as a tuple
        def forward(self, x):
            return core(*x)

    def prep(img):
        arr = np.asarray(img)   # RGB
        image, whwh, rhw = ex.prepare(arr, cfg)
        state["hw"] = (rhw, arr.shape[:2])
        torch.manual_seed(0)
        return (torch.from_numpy(image).cuda(), torch.from_numpy(whwh).cuda(), torch.randn(1, n_prop, 4).cuda()), None

    def postp(out, _):
        lg, bx = out
        rhw, hw = state["hw"]
        return ex.postprocess(lg.float().cpu().numpy(), bx.float().cpu().numpy(), rhw, hw, n_prop)
    return dict(net=Net().eval(), prep=prep, post=postp, in_shape="1x3x800x800 (+noise 1x300x4)",
                extra="SAMPLE_STEP 1, NUM_PROPOSALS 300")


MAKERS = {"rtdetrv2": make_rtdetrv2, "rfdetr": make_rfdetr, "diffusiondet": make_diffusiondet}


# ---------------------------------------------------------------- measurement
def ctx(mode):
    return torch.inference_mode() if mode == "inference_mode" else torch.no_grad()


def stats(v):
    v = np.array(v)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "min": float(v.min()), "max": float(v.max()),
            "std": float(v.std()), "n": int(len(v))}


def measure_net(net, xs, amp, mode, warm, n):
    lat = []
    with ctx(mode), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for i in range(warm):
            net(xs[i % len(xs)])
        torch.cuda.synchronize()
        for i in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            net(xs[i % len(xs)])
            b.record()
            b.synchronize()
            lat.append(a.elapsed_time(b))
    return stats(lat)


def measure_e2e(m, imgs, amp, mode, warm, n):
    lat = []
    with ctx(mode), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for i in range(warm + n):
            img = imgs[i % len(imgs)][1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, aux = m["prep"](img)
            out = m["net"](x)
            m["post"](out, aux)
            torch.cuda.synchronize()
            if i >= warm:
                lat.append(1000 * (time.perf_counter() - t0))
    return stats(lat)


def read_smi(path):
    """(timestamp, power W, temperature C, SM clock MHz) rows of an nvidia-smi CSV."""
    rows = []
    for line in open(path):
        c = [s.strip() for s in line.split(",")]
        try:
            rows.append((dt.datetime.strptime(c[0], "%Y/%m/%d %H:%M:%S.%f"), float(c[1]), float(c[2]), float(c[3])))
        except (ValueError, IndexError):
            pass
    return rows


def sample_power(seconds, fn, skip=4.0):
    """Run fn() in a loop for ``seconds`` s with nvidia-smi every 100 ms; mean power / temperature / clock, s per iteration."""
    tmp = LATENCY_DIR / "_smi.csv"
    with open(tmp, "w") as f:
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm",
                              "--format=csv,noheader,nounits", "-lms", "100"], stdout=f)
        time.sleep(1.0)
        t_start = dt.datetime.now()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            n += 1
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        t_end = dt.datetime.now()
        time.sleep(0.3)
        p.terminate()
        p.wait()
    lo = t_start + dt.timedelta(seconds=skip)
    sel = [r for r in read_smi(tmp) if lo <= r[0] <= t_end]
    if not sel:
        return {"error": "no nvidia-smi samples"}
    return {"power_W": float(np.mean([r[1] for r in sel])), "power_max_W": float(max(r[1] for r in sel)),
            "temp_C": float(np.mean([r[2] for r in sel])), "sm_clock_MHz": float(np.mean([r[3] for r in sel])),
            "samples": len(sel), "iter_s": wall / n, "n_iter": n}


def idle_power(seconds):
    return sample_power(seconds, lambda: time.sleep(0.05), skip=2.0)


def measure_power(net, xs, amp, mode, seconds):
    cnt = [0]

    def step():
        net(xs[cnt[0] % len(xs)])
        torch.cuda.synchronize()
        cnt[0] += 1
    with ctx(mode), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(20):
            step()
        return sample_power(seconds, step)


def measure_config(m, imgs, xs, amp, mode, idle_w, a):
    c = {"net_ms": measure_net(m["net"], xs, amp, mode, a.warm, a.n),
         "e2e_ms": measure_e2e(m, imgs, amp, mode, a.warm, a.n),
         "power": measure_power(m["net"], xs, amp, mode, a.power_s)}
    p = c["power"]
    if "power_W" in p:
        p["J_img_gross"] = p["power_W"] * p["iter_s"]
        p["J_img_net"] = (p["power_W"] - idle_w) * p["iter_s"]
    return c


def run(a):
    LATENCY_DIR.mkdir(parents=True, exist_ok=True)
    imgs = square_images(a.n_img)
    print(f"{len(imgs)} square VAL images; sizes", sorted({i[1].size for i in imgs}), flush=True)
    m = MAKERS[a.model](imgs)
    with torch.no_grad():
        xs = [m["prep"](img)[0] for _, img in imgs]
    print("model loaded", m["extra"], flush=True)
    with torch.inference_mode():   # sanity check: the post-processing produces detections
        out = m["post"](m["net"](xs[0]), m["prep"](imgs[0][1])[1])
    print("post-processing OK", [tuple(o.shape) for o in out], flush=True)
    res = {"model": a.model, "input": m["in_shape"], "extra": m["extra"], "n_input_images": len(imgs),
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0), "warm": a.warm, "n": a.n, "power_s": a.power_s,
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "idle": idle_power(10), "configs": {}}
    print("idle", res["idle"], flush=True)
    for prec, amp in (("fp32", False), ("amp_fp16", True)):
        for mode in ("inference_mode", "no_grad"):
            key = f"{prec}/{mode}"
            try:
                res["configs"][key] = measure_config(m, imgs, xs, amp, mode, res["idle"]["power_W"], a)
                print(key, json.dumps(res["configs"][key])[:600], flush=True)
            except Exception as e:   # one failing configuration does not stop the others
                import traceback
                res["configs"][key] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
                print(key, "ERROR", e, flush=True)
            torch.cuda.synchronize()
    res["mem_max_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    with open(LATENCY_DIR / f"{a.model}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- report
NAMES = {"rtdetrv2": "RT-DETRv2", "rfdetr": "RF-DETR", "diffusiondet": "DiffusionDet (1 step)"}


def latency_tables(raw):
    L = []
    for kind, title in (("net_ms", "Network only (CUDA events), ms"), ("e2e_ms", "End to end (pre + network + post), ms")):
        L += [f"## {title}", "", "| Detector | Config | mean | median | p95 | FPS (1000/mean) |", "|---|---|---|---|---|---|"]
        for m, r in raw.items():
            for k, c in r["configs"].items():
                if "error" in c:
                    L.append(f"| {NAMES[m]} | {k} | error: {c['error'][:80]} | | | |")
                else:
                    s = c[kind]
                    L.append(f"| {NAMES[m]} | {k} | {s['mean']:.2f} | {s['median']:.2f} | {s['p95']:.2f} | {1000 / s['mean']:.1f} |")
        L.append("")
    return L


def power_table(raw):
    L = ["## Power and energy per image (sustained network loop, batch 1)", "",
         "| Detector | Config | mean P (W) | idle P (W) | ms/iter | J/img gross | J/img net | T (C) | SM clock (MHz) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, r in raw.items():
        for k, c in r["configs"].items():
            p = c.get("power", {})
            if "power_W" in p:
                L.append(f"| {NAMES[m]} | {k} | {p['power_W']:.0f} | {r['idle']['power_W']:.0f} | {1000 * p['iter_s']:.2f} | "
                         f"{p['J_img_gross']:.2f} | {p['J_img_net']:.2f} | {p['temp_C']:.0f} | {p['sm_clock_MHz']:.0f} |")
    return L


def report(a):
    """Merge the per-model JSON files of ``--raw`` into latency_5090.md and latency_5090.json in ``--out-dir``."""
    out = Path(a.out_dir)
    raw = {}
    for m in MAKERS:
        if (Path(a.raw) / f"{m}.json").exists():
            with open(Path(a.raw) / f"{m}.json", encoding="utf8") as f:
                raw[m] = json.load(f)
    with open(out / "latency_5090.json", "w", encoding="utf8") as f:
        json.dump(raw, f, indent=1, ensure_ascii=False)
    r0 = next(iter(raw.values()))
    L = ["# Latency on the RTX 5090 (PyTorch, batch 1)", "", "Generated by `latency_5090.py --report`; data in `latency_5090.json`.", "",
         f"torch {r0['torch']}, CUDA {r0['cuda']}, cuDNN {r0['cudnn']}, {r0['gpu']}; TF32 matmul {r0['tf32_matmul']}, "
         f"TF32 cuDNN {r0['tf32_cudnn']}. {r0['warm']} warm-up iterations and {r0['n']} measurements.", ""]
    L += latency_tables(raw) + power_table(raw)
    L += ["", "Inputs: " + "; ".join(f"{NAMES[m]} {r['input']} ({r['extra']})" for m, r in raw.items()), ""]
    (out / "latency_5090.md").write_text("\n".join(L), encoding="utf8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=list(MAKERS))
    ap.add_argument("--report", action="store_true", help="merge the per-model JSON files into out/latency_5090.*")
    ap.add_argument("--raw", default=str(LATENCY_DIR), help="folder with the per-model JSON files (--report)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "out"), help="report folder (--report)")
    ap.add_argument("--n-img", type=int, default=16)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--power-s", type=float, default=20)
    a = ap.parse_args()
    if not a.report and not a.model:
        ap.error("--model is required unless --report is given")
    report(a) if a.report else run(a)


if __name__ == "__main__":
    main()
