# DenSea benchmark

Code to reproduce the experiments of *When Does Synthetic Data Help Underwater Debris Detection? Gap Metrics, Data
Regimes and Detector Paradigms* (manuscript in preparation, Neurocomputing). The study trains three detectors
(RT-DETRv2, RF-DETR and DiffusionDet) on [DenSea-Aggregated](#data) with real data only and with two kinds of
synthetic data, evaluates them with a group bootstrap, relates the result to gap metrics between synthetic and real
objects, and deploys the models on a Jetson Orin Nano Super.

This repository holds everything that is not specific to one detector: the group partition, the training sets of
every strategy, the launchers, the common evaluation, the gap analysis and the deployment tools. It contains no
images, annotations or weights.

## Related repositories

| Repository | Role | License |
|---|---|---|
| [densea-rtdetrv2](https://github.com/asferrer/densea-rtdetrv2) | RT-DETRv2 training and export | Apache-2.0 |
| [densea-rfdetr](https://github.com/asferrer/densea-rfdetr) | RF-DETR training | Apache-2.0 |
| [densea-diffusiondet](https://github.com/asferrer/densea-diffusiondet) | DiffusionDet training | CC BY-NC 4.0 (upstream license) |
| [synthetic-dataset-generator](https://github.com/asferrer/synthetic-dataset-generator) | Copy-paste synthetic data generator | Apache-2.0 |
| [annotation-tool](https://github.com/asferrer/annotation-tool) | Annotation tool used to label the data | Apache-2.0 |

This repository is Apache-2.0. The detectors run in the containers of their repositories; DiffusionDet keeps the
non-commercial license of its upstream project, so models trained with it inherit that restriction.

## Contents

```
common/          paths.py: every location is an environment variable (DATA_ROOT, RUNS_DIR, ...)
split_group/     group partition: grouping, stratified assignment, verification; out/ holds the released partition
prepare/         strategies (Real-Only, Standard, Minority-Focus, Inverse-Proportional) and experiment registry
synthetic/       requests, merge and checks of the synthetic batches (generation itself is in the generator repo)
configs/         common training configuration, RT-DETRv2 config generation, DiffusionDet backbone variants
run/             launchers (run inside the containers) and host runners: run_phase.py, predict.py, up_containers.py
eval/            COCO tools and the common TEST evaluation
analysis/        group-bootstrap analyses (full_val, proxy), box-size bias and gap metrics (synth_quality)
deploy_jetson/   ONNX export, INT8 calibration, TensorRT engines, VAL accuracy, latency and energy on the Jetson
docs/            how the synthetic batches are produced
tests/           unit tests (CPU only, no data needed)
```

## Installation

Requirements: Python 3.10+, Docker with the NVIDIA Container Toolkit and an NVIDIA GPU for training (the study used an
RTX 5090 with 32 GB), and a Jetson Orin Nano Super with JetPack 6.2.1 for the deployment part. The analysis scripts
run on the CPU.

```bash
git clone https://github.com/asferrer/densea-benchmark.git && cd densea-benchmark
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # partition, preparation and bootstrap analyses
pip install -r requirements-gap.txt        # optional: gap metrics and image descriptors (PyTorch, transformers)
pip install -r requirements-deploy.txt     # optional: ONNX export and evaluation
pip install -r requirements-dev.txt        # tests (also installs the two optional sets: some tests import torch and onnx)
python -m pytest -q                        # CPU tests
```

Check out the three detector repositories next to each other and copy `run/env.example` to your shell:

```bash
export DATA_ROOT=/path/to/densea-aggregated   # package of the Zenodo record, see below
export KIT_DIR=$PWD
export RUNS_DIR=/path/to/runs                 # weights, logs, predictions
export REPOS_DIR=/path/to/repos               # contains densea-rtdetrv2, densea-rfdetr, densea-diffusiondet
python run/up_containers.py                   # builds on top of the compose file of each detector repository
```

`run/up_containers.py` applies `run/compose/<detector>.override.yml`, which only adds the mount `/kit` (this
repository). The compose files of the detector repositories mount `DATA_ROOT` as `/data` (read-only) and `RUNS_DIR`
as `/runs`.

## Data

**DenSea-Aggregated: A Group-Aware Benchmark for Underwater Debris Detection** (Zenodo, DOI at release) unifies Ocean
Garbage, Neural Ocean and CleanSea in eight classes (Bottle, Can, Fishing_Net, Glove, Mask, Metal_Debris,
Plastic_Debris, Tire) and ships the group partition used here: 10,245 / 2,185 / 2,207 images
(train / val / test), 21,123 / 4,554 / 4,533 boxes, 995 / 204 / 217 groups (1,416 in total). Download the record
into `DATA_ROOT` and restore the CleanSea images with the script of the record (`tools/rebuild_images.py`), then check
it:

```bash
python tools/verify.py                       # inside the dataset package: checksums, counts, images present
python split_group/verify_groups.py          # no group, name key or CleanSea dive crosses a partition
```

The partition is part of the dataset. `split_group/make_split_group.py` rebuilds it from the descriptors of the
images (`split_group/features.py`), see `split_group/out/REPORT.md` for the statistics; `split_group/out/split_ids.json`
and `groups.json` are the released assignment.

**Synthetic data.** Two batches are generated with the
[synthetic-dataset-generator](https://github.com/asferrer/synthetic-dataset-generator) from the training split only and
copied under `DATA_ROOT/synthetic/`: a *clean-background* batch (objects pasted on debris-free underwater
backgrounds, label `v9`) and a *copy-paste* batch (objects pasted on real training images, label `v11`). The commands,
seeds and checks are in [`docs/synthetic_batches.md`](docs/synthetic_batches.md); the batches are not redistributed.

## Strategies and experiment ids

| Strategy | Definition |
|---|---|
| Real-Only | the real training images (`exp07_real100`) |
| Standard | real + the whole synthetic batch |
| Minority-Focus (MF) | real + synthetic images that bring Can and Metal_Debris up to 3,500 boxes |
| Inverse-Proportional (IP) | real + synthetic images that bring every class up to 4,500 boxes |

Experiment ids (`prepare/experiments.py`) combine a real fraction, a strategy and a batch, for example
`exp08_real_synth_minor` (Minority-Focus with the clean-background batch) and `exp06v11_synth100_real100`
(Standard with the copy-paste batch). Suffixes `_s2` and `_s3` are training seeds 22 and 23. Phases group the runs
(`python run/run_phase.py --help`).

## Reproduction

All commands run from the repository root. `<det>` is `rtdetrv2`, `rfdetr` or `diffusiondet`.

1. **Training sets.** `python prepare/prepare_strategies.py --lot v9` and `--lot v11` (after the batches are in place),
   then `prepare_proxy.py`, `prepare_series.py` and `copy_seed_runs.py --phase 6|8|9` for the transfer check, the
   incremental series and the extra seeds. Output: `prepare/out/<exp>/` (git-ignored).
2. **RT-DETRv2 configs.** `python configs/rtdetrv2/make_configs.py`.
3. **Training.** `python run/run_phase.py --phase <n> --detector <det> --gpu-guard`. Training never touches TEST
   (`--no-test` is implied for phases 4 and above; pass `--no-test` for the others). `--gpu-guard` runs
   `run/gpu_guard/gpu_monitor.py`, which stops the run if the GPU is lost or exceeds 83 C or the power limit.
   Main arms at 100 % real data: phase 1 (Real-Only and Minority-Focus), 5 (copy-paste Standard), 6 and 9 (seeds 22
   and 23). Configuration and stopping rule: [`configs/README.md`](configs/README.md).
4. **Predictions.** `python run/predict.py --detector <det> --split val --main-arms` (CPU by default). TEST once,
   after everything is frozen: `ALLOW_TEST=1 python run/predict.py --detector <det> --split test --main-arms`.
5. **Tables and figures**, see the next section.

### From the article to the command

| Item | Command | Output |
|---|---|---|
| Class distribution (`tab:class_distribution`) | `python split_group/make_split_group.py` | `split_group/out/REPORT.md` |
| Volumes of the training sets (`tab:synth_volumes`) | `python prepare/prepare_strategies.py --lot v9` / `--lot v11` | `prepare/reference/SUMMARY*.json` |
| Transfer check (`tab:transfer_check`, `tab:transfer_diff`) | `python analysis/proxy/px_val_bootstrap.py` | `analysis/proxy/out/px_val.md` |
| Gap along generator variants (`tab:gap_series`) | `python analysis/synth_quality/variant_pilot.py --stage all --baseline` | `analysis/synth_quality/out/variant_pilot.md` |
| Evaluator panel (`tab:panel_pilot`, `tab:panel`, `fig:ap_vs_gap`) | `python analysis/synth_quality/gap_panel.py <features|analyze|validate|report>`; `gap_panel_task.py` for the RT-DETRv2 evaluator | `out/gap_panel.md`, `gap_panel_task.md`, `gap_panel_validation.json` |
| Generator selection (`fig:generator_selection`) | `python analysis/synth_quality/gap_panel_drop.py`, `gap_panel_batch.py` | `out/gap_panel_drop.md`, `gap_panel_batch.md` |
| RT-DETRv2 with 100 % real data on VAL (`tab:full_real_val`, `fig:learning_full_real`) | `python analysis/full_val/full_val_bootstrap.py`; `extract_curves.py` | `analysis/full_val/out/full_val.md`, `curves_full.csv` |
| RF-DETR and DiffusionDet on VAL (`tab:replication_val`) | `python analysis/full_val/replica_bootstrap.py --det rfdetr` (and `diffusiondet`) | `out/replica_<det>.md` |
| Three seeds on VAL | `python analysis/full_val/seeds3_bootstrap.py --det <det>` | `out/seeds3_<det>.md` |
| **TEST results** (`tab:ablation_data`, `tab:diffdet_results`) | `ALLOW_TEST=1 python analysis/full_val/seeds3_bootstrap.py --det <det> --split test` | `out/seeds3_<det>_test.md` |
| Per class, size and errors on TEST (`tab:error_analysis_real_fraction`, `fig:confusion_test`) | `ALLOW_TEST=1 python analysis/full_val/test_analysis.py` | `out/test_analysis.md` |
| DiffusionDet sampling steps | `python run/predict.py --detector diffusiondet --split val --main-arms --steps 2 --device cuda` (also 1 and 4); `python analysis/full_val/dd_steps_bootstrap.py` | `out/dd_steps.md` |
| DiffusionDet backbones (R-101, Swin-T/S/B) | `python run/launch_dd.py --cfg ... --tag <backbone>` in the container, `python run/predict.py --detector diffusiondet --backbones res101 swintiny swinsmall swinbase --device cuda`, then `python analysis/full_val/dd_backbones_bootstrap.py` | `analysis/full_val/out/dd_backbones.md` |
| Incremental series (`tab:incremental`, `fig:incremental_series`) | `python analysis/full_val/series_bootstrap.py` | `out/series.md`, `series.json` |
| Box-size bias | `python analysis/box_bias/box_bias_val.py` | `analysis/box_bias/out/box_bias_val.md` |
| Jetson accuracy, latency, energy (`tab:jetson_accuracy`, `tab:jetson_modes`, `fig:jetson_ap_latency`) | see [`deploy_jetson/README.md`](deploy_jetson/README.md) | `deploy_jetson/out/` |
| Latency on a desktop GPU | `python deploy_jetson/latency_5090.py --model <det>` in each container, then `--report` | `deploy_jetson/out/latency_5090.md` |

Scripts that read predictions find them under `$RUNS_DIR/predictions/<det>/<split>/`. The `out/` folders contain the
reference results of the study; running a script overwrites them with your results.

## Expected results

AP@[.5:.95] (%) of the three arms trained with 100 % of the real data, mean of three training seeds (21, 22, 23), with
the 95 % CI of the group bootstrap (B = 2,000). Full tables, AP50, AP75 and paired differences:
`analysis/full_val/out/seeds3_<det>.md` (VAL, 2,185 images, 204 groups) and `seeds3_<det>_test.md` (TEST, 2,207
images, 217 groups).

| Detector | Arm | VAL AP | TEST AP |
|---|---|---|---|
| RT-DETRv2 | Real-Only | 42.86 [38.19; 48.16] | 43.42 [40.11; 48.67] |
| | Minority-Focus (clean backgrounds) | 43.75 [39.37; 49.20] | 44.00 [40.77; 49.06] |
| | Standard (copy-paste) | 42.47 [37.86; 47.51] | 43.31 [40.03; 48.18] |
| RF-DETR | Real-Only | 43.52 [39.44; 49.76] | 44.47 [40.74; 50.35] |
| | Minority-Focus (clean backgrounds) | 42.96 [38.97; 48.74] | 43.11 [39.88; 48.01] |
| | Standard (copy-paste) | 43.36 [39.31; 48.16] | 43.59 [40.67; 48.13] |
| DiffusionDet | Real-Only | 33.30 [30.11; 37.81] | 34.11 [31.83; 38.08] |
| | Minority-Focus (clean backgrounds) | 35.15 [31.56; 39.86] | 36.05 [33.09; 40.49] |
| | Standard (copy-paste) | 35.29 [31.38; 40.15] | 35.41 [32.92; 39.21] |

Paired differences of the seed means (AP, points, 95 % CI): on TEST only DiffusionDet Minority-Focus minus Real-Only
excludes zero (+1.93 [+0.09; +3.58]); RT-DETRv2 Minority-Focus minus Standard is +0.69 [-0.21; +1.96]. On VAL,
RT-DETRv2 Minority-Focus minus Standard is +1.28 [+0.21; +3.67]. Jetson results (VAL, 15 W): RT-DETRv2 INT8 mixed
43.60 AP at 35.1 ms and 359 mJ per image; RF-DETR INT8 43.31 at 23.6 ms and 275 mJ; DiffusionDet mixed 35.44 at
115.2 ms and 1,283 mJ (see `deploy_jetson/README.md`).

## Tests

`python -m pytest -q` runs the unit tests (partition, strategies, bootstrap, runners, deployment helpers). They need
neither GPU, Docker nor data.

## Citation

See `CITATION.cff`. If you use the dataset, cite the Zenodo record of DenSea-Aggregated as well.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Data and third-party models keep their own licenses: the annotations,
groups and partition of DenSea-Aggregated are CC BY 4.0, CleanSea images are subject to the terms of their source,
and DiffusionDet is CC BY-NC 4.0.
