"""Start (or stop) the training containers with the benchmark mounts (/data, /kit, /runs).

    python run/up_containers.py [--detectors rtdetrv2 rfdetr diffusiondet] [--down]

Applies ``run/compose/<detector>.override.yml`` on top of the compose file of each detector repository, which is
looked up in ``$REPOS_DIR`` (see ``run/containers.py``). Required environment: ``DATA_ROOT``, ``KIT_DIR``,
``RUNS_DIR`` and ``REPOS_DIR`` (see ``run/env.example``).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containers import COMPOSE, DETECTORS  # noqa: E402

OVERRIDES = Path(__file__).resolve().parent / "compose"


def require_env(names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit("define " + ", ".join(missing) + " (see run/env.example)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detectors", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    ap.add_argument("--down", action="store_true", help="stop and remove the containers")
    a = ap.parse_args()
    require_env(["DATA_ROOT", "KIT_DIR", "RUNS_DIR", "REPOS_DIR"])
    Path(os.environ["RUNS_DIR"]).mkdir(parents=True, exist_ok=True)
    if not a.down and subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL) != 0:
        sys.exit("nvidia-smi fails: the GPU is not available")
    for det in a.detectors:
        compose = Path(os.environ["REPOS_DIR"]) / COMPOSE[det]
        cmd = ["docker", "compose", "-f", str(compose), "-f", str(OVERRIDES / f"{det}.override.yml")]
        cmd += ["down"] if a.down else ["up", "-d"]
        if subprocess.call(cmd) != 0:
            sys.exit(f"docker compose failed for {det}")
    subprocess.call(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"])


if __name__ == "__main__":
    main()
