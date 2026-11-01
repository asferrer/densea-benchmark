"""Train and/or evaluate one RT-DETRv2 experiment INSIDE the densea-rtdetrv2 container.

    docker exec -it densea_rtdetrv2 python /kit/run/launch_rt.py --exp exp07_real100 [--eval-only]

Training runs ``tools/train.py`` with ``/runs/configs/rtdetrv2/<exp>.yml`` (written by
``configs/rtdetrv2/make_configs.py``; validation on the VAL partition; ``best.pth`` is the EMA checkpoint with the highest VAL AP; early stopping after 20 epochs without improvement) and
resumes from ``last.pth`` when it exists. Evaluation runs ``tools/eval.py`` with ``<exp>_eval.yml`` (TEST partition)
on ``best.pth``; predictions are written to ``/runs/rtdetrv2/<exp>/eval_test/best/coco_predictions.json``.
"""
import argparse
import os
import subprocess
import sys

# faster-coco-eval gives the same results as pycocotools (difference <= 2e-16) and only speeds up the VAL
# evaluation during training. FAST_COCOEVAL=0 disables it.
os.environ.setdefault("FAST_COCOEVAL", "1")

REPO = "/app/RT-DETR"
CFG = "/runs/configs/rtdetrv2"
sys.path.insert(0, "/kit")
from prepare.experiments import EXPERIMENTS  # noqa: E402


def run(cmd, log):
    """Run a command inside the repository, echoing and appending its output to ``log``."""
    print(" ".join(cmd), flush=True)
    with open(log, "a", encoding="utf8") as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip the TEST evaluation after training")
    args = ap.parse_args()
    out = f"/runs/rtdetrv2/{args.exp}"
    os.makedirs(out, exist_ok=True)
    if not args.eval_only:
        seed = EXPERIMENTS.get(args.exp, {}).get("train_seed", 21)
        cmd = ["python", "tools/train.py", "-c", f"{CFG}/{args.exp}.yml", "-d", "cuda", "--seed", str(seed),
               "--use-amp"]
        if os.path.exists(f"{out}/last.pth"):
            cmd += ["-r", f"{out}/last.pth"]
        if run(cmd, f"{out}/train.log") != 0:
            sys.exit("training failed")
    if args.no_test:
        return
    cmd = ["python", "tools/eval.py", "-c", f"{CFG}/{args.exp}_eval.yml", "-r", f"{out}/best.pth",
           "--output_dir", f"{out}/eval_test", "-d", "cuda", "--coco_eval"]
    sys.exit(run(cmd, f"{out}/eval_test.log"))


if __name__ == "__main__":
    main()
