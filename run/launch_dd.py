"""Train and/or evaluate one DiffusionDet experiment INSIDE the densea-diffusiondet container.

    docker exec -it densea_diffusiondet python /kit/run/launch_dd.py --exp exp07_real100 [--eval-only]

Common stopping rule: at most 200 equivalent epochs (``MAX_ITER = 200 * ceil(N / batch)``, batch 8), validation
every equivalent epoch, ``model_best.pth`` = highest ``bbox/AP`` (EMA weights) on VAL, and stop after 20
evaluations (= 20 epochs) without an improvement of 0.1 AP points (= 0.001). The final evaluation uses
``model_best.pth`` (EMA) on TEST; predictions are written to
``/runs/diffusiondet/<exp>/eval_test/inference/densea_test/coco_instances_results.json``.
Periodic checkpoints are saved every 10 epochs; those of every 20 epochs and the last one are kept
(``run/launch_diag.py`` evaluates them).
"""
import argparse
import json
import math
import os
import subprocess
import sys

REPO = "/app/densea-diffusiondet"
CFG = f"{REPO}/configs/diffdet.densea.res50.yaml"
PATIENCE_KEY = "DENSEA.EARLY_STOP_PATIENCE"
KEEP_EVERY_KEY = "DENSEA.CHECKPOINT_KEEP_EVERY"


def schedule(exp, epochs, batch, eval_every):
    """Iteration counts derived from the number of training images of ``exp``."""
    n = len(json.load(open(f"/kit/prepare/out/{exp}/train.json"))["images"])
    ipe = math.ceil(n / batch)
    return {"SOLVER.MAX_ITER": epochs * ipe, "TEST.EVAL_PERIOD": eval_every * ipe,
            "SOLVER.CHECKPOINT_PERIOD": 10 * ipe, KEEP_EVERY_KEY: 20 * ipe,
            "SOLVER.IMS_PER_BATCH": batch, "_images": n}


def env_for(exp):
    """Environment that points the training script to the JSON files and image root of ``exp``.

    ``DENSEA_TEST_WORKERS=4`` gives the same predictions as 0 workers and a faster evaluation.
    ``DENSEA_FAST_OPT=1`` enables the faster optimisation path validated on a long run (AP on VAL within the
    run-to-run noise).
    """
    d = f"/kit/prepare/out/{exp}"
    return {**os.environ, "DENSEA_TEST_WORKERS": os.environ.get("DENSEA_TEST_WORKERS", "4"),
            "DENSEA_FAST_OPT": os.environ.get("DENSEA_FAST_OPT", "1"),
            "DENSEA_TRAIN_JSON": f"{d}/train.json", "DENSEA_VALID_JSON": f"{d}/val.json",
            "DENSEA_TEST_JSON": f"{d}/test.json", "DENSEA_IMAGE_ROOT": "/data"}


def train_seed(exp):
    sys.path.insert(0, "/kit")
    from prepare.experiments import EXPERIMENTS
    return EXPERIMENTS.get(exp, {}).get("train_seed", 21)


def opts_list(d):
    """Flatten a dict into the ``KEY VALUE`` list expected by the training script (keys starting with _ are skipped)."""
    return [str(x) for k, v in d.items() if not k.startswith("_") for x in (k, v)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20, help="evaluations without improvement")
    ap.add_argument("--cfg", default=CFG, help="YAML of another backbone (configs/diffusiondet/*.yaml)")
    ap.add_argument("--tag", help="suffix of the output folder: /runs/diffusiondet/<exp>__<tag>")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip the TEST evaluation after training")
    args = ap.parse_args()
    out = f"/runs/diffusiondet/{args.exp}" + (f"__{args.tag}" if args.tag else "")
    os.makedirs(out, exist_ok=True)
    sch = schedule(args.exp, args.epochs, args.batch, args.eval_every)
    sch[PATIENCE_KEY] = args.patience
    sch["SEED"] = train_seed(args.exp)  # 21 in the YAML; 22/23 for the extra-seed runs (prepare/experiments.py)
    json.dump(sch, open(f"{out}/schedule.json", "w"), indent=1)
    base = ["python", "train_net_densea.py", "--num-gpus", "1", "--config-file", args.cfg]
    if not args.eval_only:
        cmd = base + ["--resume", "OUTPUT_DIR", out] + opts_list(sch)
        print(" ".join(cmd), flush=True)
        if subprocess.call(cmd, cwd=REPO, env=env_for(args.exp)) != 0:
            sys.exit("training failed")
    if args.no_test:
        return
    cmd = base + ["--eval-only", "MODEL.WEIGHTS", f"{out}/model_best.pth", "DATASETS.TEST", '("densea_test",)',
                  "OUTPUT_DIR", f"{out}/eval_test"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO, env=env_for(args.exp)))


if __name__ == "__main__":
    main()
