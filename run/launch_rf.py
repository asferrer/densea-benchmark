"""Train and/or evaluate one RF-DETR (Base) experiment INSIDE the densea-rfdetr container.

    docker exec -it densea_rfdetr python /kit/run/launch_rf.py --exp exp07_real100 [--eval-only]

Hyper-parameters come from ``densea/hyperparams.py`` of the RF-DETR repository (base variant:
lr 5e-5, encoder lr 5e-6, weight decay 0.05, 200 epochs, 6 warm-up epochs, cosine schedule, batch 8 x 4 gradient
accumulation, EMA 0.9998, patience 20 / min_delta 0.001, multi-scale, underwater augmentation, resolution 560).
Three settings align the selection criterion with the other detectors: only the EMA model is evaluated
(``eval_regular_model=False``), early stopping uses the EMA (``early_stopping_use_ema=True``), and the evaluated
checkpoint is ``checkpoint_best_ema.pth`` (highest VAL AP of the EMA). TEST is not touched during training
(``run_test=False``); predictions are written by ``densea/dump_test_predictions.py`` to
``/runs/rfdetr/<exp>/eval_test/coco_predictions.json``.
"""
import argparse
import os
import subprocess
import sys

# faster-coco-eval gives the same results as pycocotools (difference <= 2e-16) and only speeds up the VAL
# evaluation during training. FAST_COCOEVAL=0 disables it.
os.environ.setdefault("FAST_COCOEVAL", "1")

REPO = "/app/densea-rfdetr"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/src")


def exp_info(exp):
    """Training seed and data folder (seed 22/23 runs reuse the data of the seed 21 run)."""
    sys.path.insert(0, "/kit")
    from prepare.experiments import EXPERIMENTS
    e = EXPERIMENTS.get(exp, {})
    return e.get("train_seed", 21), e.get("data_from", exp)


def hyperparams(exp, out):
    from densea.hyperparams import get_hyperparams
    seed, data = exp_info(exp)
    hp = get_hyperparams(f"/kit/prepare/out/rfdetr/{data}", out, seed, f"{exp}_base")
    if os.path.exists(f"{out}/checkpoint.pth"):
        hp["resume"] = f"{out}/checkpoint.pth"
    return hp


def train(exp, out):
    import rfdetr
    hp = hyperparams(exp, out)
    print({k: v for k, v in hp.items() if k != "aug_config"}, flush=True)
    rfdetr.RFDETRBase().train(**hp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip the TEST evaluation after training")
    args = ap.parse_args()
    out = f"/runs/rfdetr/{args.exp}"
    os.makedirs(out, exist_ok=True)
    if not args.eval_only:
        train(args.exp, out)
    if args.no_test:
        return
    cmd = ["python", "densea/dump_test_predictions.py", "--checkpoint", f"{out}/checkpoint_best_ema.pth",
           "--output-dir", f"{out}/eval_test", "--dataset-dir", f"/kit/prepare/out/rfdetr/{exp_info(args.exp)[1]}"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
