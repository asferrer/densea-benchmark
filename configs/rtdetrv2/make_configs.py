"""Generate the RT-DETRv2 YAML files of every experiment with the generator of the detector repository.

For each experiment two files are written inside the RT-DETRv2 container (the includes of the files point to the
repository at ``/app/RT-DETR``), under ``$RUNS_DIR/configs/rtdetrv2/``:

  <exp>.yml        training; the validation dataloader reads the VAL partition (model selection)
  <exp>_eval.yml   final evaluation; the validation dataloader reads the TEST partition

The common settings (AdamW 5e-5 / backbone 5e-6, weight decay 0.05, cosine schedule, EMA 0.9998, batch 32,
multi-scale 640-800, early stopping with patience 20 and min_delta 0.001 on the VAL AP) live in the template
``configs/densea/rtdetrv2_r50vd_densea.template.yml`` of the detector repository.

    python configs/rtdetrv2/make_configs.py [--container densea_rtdetrv2]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prepare.experiments import EXPERIMENTS  # noqa: E402

CONFIG_DIR = "/runs/configs/rtdetrv2"


def command(container, exp):
    """``docker exec`` call of tools/make_densea_config.py for one experiment (paths as seen in the container)."""
    data = f"/kit/prepare/out/{exp}"
    return ["docker", "exec", container, "python", "tools/make_densea_config.py", "--name", exp,
            "--train-json", f"{data}/train.json", "--val-json", f"{data}/val.json",
            "--test-json", f"{data}/test.json", "--image-root", "/data",
            "--output-dir", f"/runs/rtdetrv2/{exp}", "--out-dir", CONFIG_DIR]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--container", default="densea_rtdetrv2")
    a = ap.parse_args()
    for exp in EXPERIMENTS:
        subprocess.run(command(a.container, exp), check=True, stdout=subprocess.DEVNULL)
    print("wrote", 2 * len(EXPERIMENTS), "files to", CONFIG_DIR, "(inside the container)")


if __name__ == "__main__":
    main()
