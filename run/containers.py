"""Constants and helpers shared by the host-side runners (``run_phase.py``, ``predict.py``, ``up_containers.py``).

The three detectors run in the containers defined by the sibling repositories. Their compose files mount ``/data``
(DATA_ROOT, read-only) and ``/runs`` (RUNS_DIR); ``run/compose/*.override.yml`` adds ``/kit`` (this repository,
read-only).
"""
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from common import paths  # noqa: E402,F401
from prepare.experiments import EXPERIMENTS  # noqa: E402,F401

DETECTORS = ("rtdetrv2", "rfdetr", "diffusiondet")
CONTAINER = {"rtdetrv2": "densea_rtdetrv2", "rfdetr": "densea_rfdetr", "diffusiondet": "densea_diffusiondet"}
LAUNCHER = {"rtdetrv2": "launch_rt.py", "rfdetr": "launch_rf.py", "diffusiondet": "launch_dd.py"}
# Compose file of each sibling repository, relative to the folder that holds the checkouts (REPOS_DIR).
COMPOSE = {"rtdetrv2": "densea-rtdetrv2/docker-compose.yml",
           "rfdetr": "densea-rfdetr/docker-compose.yml",
           "diffusiondet": "densea-diffusiondet/docker-compose.yml"}


def experiments_of_phase(phase, only=()):
    """Experiment ids of a phase, optionally restricted to ``only``."""
    return [k for k, v in EXPERIMENTS.items() if v["phase"] == phase and (not only or k in only)]


def docker_exec(container, args, env=None, workdir=None, dry_run=False):
    """Run ``docker exec`` and return its exit code (the command is printed instead when ``dry_run``)."""
    cmd = ["docker", "exec"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    if workdir:
        cmd += ["-w", workdir]
    cmd += [container] + list(args)
    print(" ".join(cmd), flush=True)
    return 0 if dry_run else subprocess.call(cmd)
