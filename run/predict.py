"""Dump COCO detections of the best checkpoint of each run on VAL or TEST (the input of the bootstrap analyses).

    python run/predict.py --detector rtdetrv2 --split val --exps exp07_real100 exp08_real_synth_minor
    python run/predict.py --detector diffusiondet --split val --phase 9 --steps 1 2 4
    ALLOW_TEST=1 python run/predict.py --detector rfdetr --split test --main-arms
    python run/predict.py --detector diffusiondet --backbones res101 swintiny swinsmall swinbase --device cuda

Runs ``docker exec`` into the detector container. Inference is done on the CPU by default (``--device cuda`` for the
GPU) so that the numbers do not depend on the GPU model and the card stays free for training. Output files are
``$RUNS_DIR/predictions/<detector>/<split>/<exp>_<split>_preds.json`` (``steps<k>/`` sub-folder for DiffusionDet
with ``--steps k`` sampling steps); existing files are skipped.

TEST is evaluated once, with the frozen configuration: set ALLOW_TEST=1 to confirm.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containers import CONTAINER, DETECTORS, EXPERIMENTS, docker_exec, experiments_of_phase, paths  # noqa: E402

MAIN_ARMS = ("exp07_real100", "exp08_real_synth_minor", "exp06v11_synth100_real100")
SEED_SUFFIXES = ("", "_s2", "_s3")
DD_RESULT = {"val": "densea_valid", "test": "densea_test"}


def container_dir(detector, split, steps):
    sub = f"/steps{steps}" if steps else ""
    return f"/runs/predictions/{detector}/{split}{sub}"


def host_file(detector, split, exp, steps):
    sub = f"steps{steps}" if steps else ""
    return paths.predictions_dir(detector, split) / sub / f"{exp}_{split}_preds.json"


def data_of(exp):
    """Experiment whose data folder is used (extra-seed runs share the data of the seed 21 run)."""
    return EXPERIMENTS.get(exp, {}).get("data_from", exp)


def rtdetrv2_cmd(exps, split, out, threads):
    return ["nice", "-n", "19", "python", "/kit/analysis/box_bias/infer_val_cpu.py", "--split", split, "--exps",
            *exps, "--out", out, "--threads", str(threads)]


def rfdetr_cmd(exp, split, out, device):
    work = f"{out}/_work/{exp}"
    return ["sh", "-c", f"python densea/dump_test_predictions.py --checkpoint "
            f"/runs/rfdetr/{exp}/checkpoint_best_ema.pth --output-dir {work} --dataset-dir "
            f"/kit/prepare/out/rfdetr/{data_of(exp)} --split {split} --device {device} --num-workers 2 && "
            f"cp {work}/coco_predictions.json {out}/{exp}_{split}_preds.json"]


def diffusiondet_cmd(exp, split, out, device, steps):
    work = f"{out}/_work/{exp}"
    opts = f"MODEL.DEVICE {device} " if device == "cpu" else ""
    opts += f"MODEL.DiffusionDet.SAMPLE_STEP {steps} " if steps else ""
    name = DD_RESULT[split]
    return ["sh", "-c", f"python train_net_densea.py --num-gpus 1 --config-file "
            f"/app/densea-diffusiondet/configs/diffdet.densea.res50.yaml --eval-only {opts}MODEL.WEIGHTS "
            f"/runs/diffusiondet/{exp}/model_best.pth DATASETS.TEST '(\"{name}\",)' OUTPUT_DIR {work} && "
            f"cp {work}/inference/{name}/coco_instances_results.json {out}/{exp}_{split}_preds.json"]


def dd_env(exp, threads):
    d = f"/kit/prepare/out/{data_of(exp)}"
    return {"DENSEA_TEST_WORKERS": 2, "DENSEA_FAST_OPT": 1, "OMP_NUM_THREADS": threads,
            "DENSEA_TRAIN_JSON": f"{d}/train.json", "DENSEA_VALID_JSON": f"{d}/val.json",
            "DENSEA_TEST_JSON": f"{d}/test.json", "DENSEA_IMAGE_ROOT": "/data"}


def run_exps(a, exps):
    container, env = CONTAINER[a.detector], {}
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if a.split == "test":
        env["ALLOW_TEST"] = 1
    out = container_dir(a.detector, a.split, a.steps)
    if a.detector == "rtdetrv2":
        todo = [e for e in exps if not host_file(a.detector, a.split, e, a.steps).exists()]
        return docker_exec(container, rtdetrv2_cmd(todo, a.split, out, a.threads), env, dry_run=a.dry_run) if todo else 0
    for exp in exps:
        if host_file(a.detector, a.split, exp, a.steps).exists():
            print(exp, "already done")
            continue
        if a.detector == "rfdetr":
            cmd, workdir, extra = rfdetr_cmd(exp, a.split, out, a.device), "/app/densea-rfdetr", {"OMP_NUM_THREADS": a.threads}
        else:
            cmd, workdir = diffusiondet_cmd(exp, a.split, out, a.device, a.steps), "/app/densea-diffusiondet"
            extra = dd_env(exp, a.threads)
        rc = docker_exec(container, cmd, {**env, **extra}, workdir, a.dry_run)
        if rc != 0:
            return rc
    return 0


def backbone_cmd(tag, out):
    """VAL inference (1 sampling step) of the Real-Only run with another DiffusionDet backbone."""
    work = f"{out}/_work/{tag}"
    return ["sh", "-c", f"python train_net_densea.py --num-gpus 1 --config-file "
            f"/kit/configs/diffusiondet/diffdet.densea.{tag}.yaml --eval-only MODEL.WEIGHTS "
            f"/runs/diffusiondet/exp07_real100__{tag}/model_best.pth MODEL.DiffusionDet.SAMPLE_STEP 1 "
            f"DATASETS.TEST '(\"densea_valid\",)' OUTPUT_DIR {work} && "
            f"cp {work}/inference/densea_valid/coco_instances_results.json {out}/{tag}_val_preds.json"]


def run_backbones(a):
    out = "/runs/predictions/diffusiondet/val/backbones"
    for tag in a.backbones:
        if (paths.predictions_dir("diffusiondet", "val") / "backbones" / f"{tag}_val_preds.json").exists():
            print(tag, "already done")
            continue
        env = dd_env("exp07_real100", a.threads)
        if a.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        rc = docker_exec(CONTAINER["diffusiondet"], backbone_cmd(tag, out), env, "/app/densea-diffusiondet", a.dry_run)
        if rc != 0:
            return rc
    return 0


def select(a):
    if a.exps:
        return a.exps
    if a.main_arms:
        return [e + s for e in MAIN_ARMS for s in SEED_SUFFIXES]
    return experiments_of_phase(a.phase)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detector", required=True, choices=DETECTORS)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--backbones", nargs="+", choices=["res101", "swintiny", "swinsmall", "swinbase"],
                    help="DiffusionDet backbone runs (exp07_real100__<tag>), VAL, 1 step; ignores --split/--exps")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--exps", nargs="+")
    group.add_argument("--phase", type=int)
    group.add_argument("--main-arms", action="store_true", help="the 3 main arms x seeds 21, 22 and 23")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--steps", type=int, default=0, help="DiffusionDet sampling steps (0 = value of the config)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    if a.backbones:
        if a.detector != "diffusiondet":
            sys.exit("--backbones only applies to diffusiondet")
        sys.exit(run_backbones(a))
    if not (a.exps or a.phase is not None or a.main_arms):
        sys.exit("give --exps, --phase, --main-arms or --backbones")
    if a.split == "test" and os.environ.get("ALLOW_TEST") != "1":
        sys.exit("TEST is evaluated once, with the frozen configuration; set ALLOW_TEST=1 to run it.")
    if a.steps and a.detector != "diffusiondet":
        sys.exit("--steps only applies to diffusiondet")
    sys.exit(run_exps(a, select(a)))


if __name__ == "__main__":
    main()
