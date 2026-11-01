"""Run the experiments of one phase for one detector, one after another, on a single GPU.

    python run/run_phase.py --phase 1 --detector rtdetrv2 [--only exp07_real100] [--eval-only] [--no-test]
                            [--diag] [--gpu-guard]

Phases (see ``prepare/experiments.py``):
  1  main strategies with 100% of the real data (Real-Only, Standard, Minority-Focus, Inverse-Proportional)
  2  incremental series (0/10/25/50/75% real data, with and without the full synthetic batch)
  3  optional extra arm (10% real data + Inverse-Proportional)
  4  transfer check of the generator (RT-DETRv2 only, VAL only)
  5  strategies with the copy-paste batch (VAL only)
  6  second seed (22) of the main arms (VAL only)
  7  incremental series with both batches at 25% and 50% real data (RT-DETRv2, VAL only)
  8  second seed of the 25% series (VAL only)
  9  third seed (23) of the main arms (VAL only)
Phases 4 and above never evaluate TEST. The progress of each phase is stored in
``$RUNS_DIR/state/phase<N>_<detector>.json`` and finished experiments are skipped when the script is launched again.

``--diag`` evaluates the intermediate checkpoints on TEST after each run (diagnostic only, ``run/launch_diag.py``).
``--gpu-guard`` runs ``run/gpu_guard/gpu_monitor.py`` next to the training and stops it when the GPU is lost or
exceeds the temperature/power limits.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containers import CONTAINER, DETECTORS, LAUNCHER, experiments_of_phase, paths  # noqa: E402

MONITOR = Path(__file__).resolve().parent / "gpu_guard" / "gpu_monitor.py"


def start_guard(log_dir, max_temp, max_power):
    """Start the GPU monitor and return its process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    alert = Path(log_dir) / "ALERT.txt"
    alert.unlink(missing_ok=True)
    return subprocess.Popen([sys.executable, str(MONITOR), "--log-dir", str(log_dir), "--interval", "5",
                             "--max-temp", str(max_temp), "--max-power", str(max_power)])


def wait_with_guard(proc, container, alert):
    """Wait for ``proc``; if the monitor writes an alert, stop the launcher and return False."""
    while proc.poll() is None:
        time.sleep(10)
        if alert is not None and alert.exists():
            print("GPU alert:", alert.read_text().strip().splitlines()[-1], flush=True)
            subprocess.call(["docker", "exec", container, "pkill", "-f", "launch_"])
            proc.terminate()
            return False
    return proc.returncode == 0


def launch(container, launcher, exp, flags, alert):
    """Run one experiment inside its container; True when it finished correctly."""
    cmd = ["docker", "exec", container, "python", f"/kit/run/{launcher}", "--exp", exp] + flags
    print(" ".join(cmd), flush=True)
    return wait_with_guard(subprocess.Popen(cmd), container, alert)


def diagnose(container, detector, exp):
    cmd = ["docker", "exec", container, "python", "/kit/run/launch_diag.py", "--det", detector, "--exp", exp]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--detector", required=True, choices=DETECTORS)
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these experiment ids")
    ap.add_argument("--eval-only", action="store_true", help="only evaluate TEST with the existing best checkpoint")
    ap.add_argument("--no-test", action="store_true", help="train without evaluating TEST")
    ap.add_argument("--diag", action="store_true", help="evaluate intermediate checkpoints on TEST afterwards")
    ap.add_argument("--gpu-guard", action="store_true")
    ap.add_argument("--max-temp", type=float, default=83)
    ap.add_argument("--max-power", type=float, default=440)
    a = ap.parse_args()
    no_test = a.no_test or a.phase >= 4
    if a.phase == 4 and (a.detector != "rtdetrv2" or a.eval_only):
        sys.exit("phase 4 trains RT-DETRv2 only and never evaluates TEST")
    if no_test and a.eval_only:
        sys.exit("--eval-only evaluates TEST and cannot be combined with --no-test or phases >= 4")
    state_file = paths.RUNS_DIR / "state" / f"phase{a.phase}_{a.detector}.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    log_dir = paths.RUNS_DIR / "gpu_logs" / f"phase{a.phase}_{a.detector}"
    guard = start_guard(log_dir, a.max_temp, a.max_power) if a.gpu_guard else None
    alert = log_dir / "ALERT.txt" if guard else None
    flags = ["--eval-only"] if a.eval_only else (["--no-test"] if no_test else [])
    try:
        for exp in experiments_of_phase(a.phase, a.only):
            if state.get(exp) == "done" and not a.eval_only:
                print(exp, "already done")
                continue
            t0 = time.time()
            ok = launch(CONTAINER[a.detector], LAUNCHER[a.detector], exp, flags, alert)
            state[exp] = "done" if ok else "failed"
            state[f"{exp}_hours"] = round((time.time() - t0) / 3600, 2)
            if ok and a.diag and not no_test and not a.eval_only:
                state[f"{exp}_diag"] = "done" if diagnose(CONTAINER[a.detector], a.detector, exp) else "failed"
            state_file.write_text(json.dumps(state, indent=1))
            if not ok:
                sys.exit(f"{a.detector} {exp} failed (see {paths.RUNS_DIR / a.detector / exp})")
    finally:
        if guard:
            guard.terminate()
    print(f"Phase {a.phase} of {a.detector} finished. State: {state_file}")


if __name__ == "__main__":
    main()
