"""Evaluate the intermediate checkpoints of a run on TEST (overfitting diagnostic).

    docker exec densea_rtdetrv2 python /kit/run/launch_diag.py --det rtdetrv2     --exp exp07_real100
    docker exec densea_rfdetr   python /kit/run/launch_diag.py --det rfdetr       --exp exp07_real100
    docker exec densea_diffusiondet python /kit/run/launch_diag.py --det diffusiondet --exp exp07_real100

Diagnostic only: the reported model is always the one selected on VAL (``best.pth``,
``checkpoint_best_ema.pth``, ``model_best.pth``); these numbers never change the selection. The checkpoints are
those of every 20 epochs and the last one, evaluated with their EMA weights like the final evaluation. Output:
``/runs/<det>/<exp>/eval_diag/<ep020|...|last>/`` with predictions in the same format as ``eval_test/``; folders
that already hold a ``DONE`` file are skipped. Curves and table: ``eval/diag_curves.py`` (on the host).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EVERY = 20


def numbered(out, pattern, base):
    """Checkpoints named ``checkpointNNNN.pth`` (0-based epoch index) whose epoch is a multiple of EVERY."""
    found = {}
    for f in glob.glob(f"{out}/{pattern}"):
        ep = int(re.search(r"(\d{4})\.pth$", f).group(1)) + 1
        if ep % EVERY == 0:
            found[f"ep{ep:03d}"] = f
    if os.path.exists(f"{out}/{base}"):
        found["last"] = f"{out}/{base}"
    return found


def rt_ckpts(out):
    import launch_rt
    return numbered(out, "checkpoint[0-9][0-9][0-9][0-9].pth", "last.pth"), launch_rt.REPO


def rf_ckpts(out):
    import launch_rf
    return numbered(out, "checkpoint[0-9][0-9][0-9][0-9].pth", "checkpoint.pth"), launch_rf.REPO


def dd_ckpts(out):
    import launch_dd
    ipe = json.load(open(f"{out}/schedule.json"))["SOLVER.CHECKPOINT_PERIOD"] // 10
    found, latest = {}, None
    for f in sorted(glob.glob(f"{out}/model_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pth")):
        it = int(re.search(r"(\d{7})\.pth$", f).group(1)) + 1
        latest = f
        if it % (EVERY * ipe) == 0:
            found[f"ep{it // ipe:03d}"] = f
    last = f"{out}/model_final.pth" if os.path.exists(f"{out}/model_final.pth") else latest
    if last and last not in found.values():
        found["last"] = last
    return found, launch_dd.REPO


def command(det, exp, ckpt, d):
    """Evaluation command (and optional environment) of one checkpoint."""
    if det == "rtdetrv2":
        import launch_rt
        return ["python", "tools/eval.py", "-c", f"{launch_rt.CFG}/{exp}_eval.yml", "-r", ckpt,
                "--output_dir", d, "-d", "cuda", "--coco_eval"], None
    if det == "rfdetr":
        return ["python", "experiments/dump_test_predictions.py", "--checkpoint", ckpt, "--ema",
                "--output-dir", d, "--dataset-dir", f"/kit/prepare/out/rfdetr/{exp}"], None
    import launch_dd
    return ["python", "train_net_densea.py", "--num-gpus", "1", "--config-file", launch_dd.CFG,
            "--eval-only", "MODEL.WEIGHTS", ckpt, "DATASETS.TEST", '("densea_test",)',
            "OUTPUT_DIR", d], launch_dd.env_for(exp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--det", required=True, choices=["rtdetrv2", "rfdetr", "diffusiondet"])
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    out = f"/runs/{args.det}/{args.exp}"
    ckpts, repo = {"rtdetrv2": rt_ckpts, "rfdetr": rf_ckpts, "diffusiondet": dd_ckpts}[args.det](out)
    if not ckpts:
        sys.exit(f"no intermediate checkpoints in {out}")
    failed = []
    for tag, ckpt in sorted(ckpts.items()):
        d = f"{out}/eval_diag/{tag}"
        if os.path.exists(f"{d}/DONE"):
            print(tag, "already evaluated", flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        cmd, env = command(args.det, args.exp, ckpt, d)
        print(tag, " ".join(cmd), flush=True)
        with open(f"{d}/diag.log", "w", encoding="utf8") as log:
            rc = subprocess.call(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc == 0:
            open(f"{d}/DONE", "w").write(os.path.basename(ckpt) + "\n")
        else:
            failed.append(tag)
    if failed:
        sys.exit(f"diagnostic failed for {', '.join(failed)} (see eval_diag/<tag>/diag.log)")


if __name__ == "__main__":
    main()
