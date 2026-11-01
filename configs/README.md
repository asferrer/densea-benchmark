# Training configuration

All strategies of a detector are trained with one common configuration; only the training set changes
(`prepare/out/<exp>/`). The three detectors share the data, the partition, the model-selection rule and the stopping
rule.

## Common rules

- **Data.** `prepare/out/<exp>/{train,val,test}.json`. VAL and TEST are the real images of the group partition; the
  training set is the real subset plus the synthetic images of the strategy. Synthetic images without annotations are
  discarded.
- **Model selection.** EMA weights with the highest AP@[.5:.95] on VAL, evaluated during training.
- **Stopping rule.** At most 200 epochs; stop when the VAL AP@[.5:.95] does not improve by more than 0.001 for 20
  epochs. DiffusionDet trains by iterations: it runs `200 x ceil(N/8)` iterations, evaluates once per equivalent epoch
  and stops after 20 evaluations without improvement.
- **Final evaluation.** TEST, with `eval/evaluate_runs.py` (pycocotools, at most 100 detections per image) and the
  group bootstrap of `analysis/full_val/` (B = 2,000).
- **Optimisation.** AdamW, learning rate 5e-5 for the heads and 5e-6 for the backbone, weight decay 0.05, cosine
  schedule with warm-up, EMA 0.9998, gradient clipping 1.0, mixed precision, seed 21 (22 and 23 for the extra seeds).
  Effective batch size 32 (RT-DETRv2, RF-DETR) and 8 (DiffusionDet).
- **Initialisation and resolution.** Native to each detector: RT-DETRv2 PResNet-50vd ImageNet, 640 (multi-scale
  640-800); RF-DETR-B COCO, 560; DiffusionDet R-50 ImageNet, 800/960. They are reported, not equalised.

| | RT-DETRv2 | RF-DETR-B | DiffusionDet R50 |
|---|---|---|---|
| File | `/runs/configs/rtdetrv2/<exp>.yml` (+ `_eval.yml` for TEST) | `run/launch_rf.py` (settings of `densea/hyperparams.py`) | `configs/diffdet.densea.res50.yaml` of densea-diffusiondet + `run/launch_dd.py` |
| Batch | 32 (drop_last) | 8 x 4 accumulation | 8, no accumulation |
| Maximum | 200 epochs | 200 epochs | 200 equivalent epochs (`200 x ceil(N/8)` iterations) |
| VAL evaluation | every epoch | every epoch (EMA only) | every equivalent epoch (EMA) |
| Patience / min_delta | 20 epochs / 0.001 | 20 epochs / 0.001 (`early_stopping_use_ema=True`) | 20 evaluations / 0.1 AP points |
| Evaluated checkpoint | `best.pth` (EMA) | `checkpoint_best_ema.pth` | `model_best.pth` (EMA) |
| Classification loss | varifocal (alpha 0.75, gamma 2) | RF-DETR default | focal (alpha 0.25, gamma 2) |
| Queries / proposals | 150 | 300 | 300 |
| Sampler | random | random | `TrainingSampler` (repeat-factor sampling off) |

## RT-DETRv2

`configs/rtdetrv2/make_configs.py` calls `tools/make_densea_config.py` of densea-rtdetrv2 inside its container for
every experiment and writes `<exp>.yml` (training, VAL) and `<exp>_eval.yml` (TEST) to `$RUNS_DIR/configs/rtdetrv2/`.
Both come from the template `configs/densea/rtdetrv2_r50vd_densea.template.yml` of that repository, which holds the
common settings. Settings that are part of the method:

- A single image root (`/data/`) with the relative paths stored in the COCO files.
- The `stop_epoch` policy (epoch 180) switches off multi-scale training and the three augmentations listed in the
  template (`RandomPhotometricDistort`, `RandomGaussianBlur`, `RandomIoUCrop`); the other transforms stay active.

## RF-DETR

`run/launch_rf.py` uses `densea/hyperparams.py` of densea-rfdetr (RF-DETR Base, underwater augmentation) with three
settings that align the selection with the other detectors: `early_stopping_use_ema=True`, `run_test=False` (TEST is evaluated afterwards with
`densea/dump_test_predictions.py`) and the evaluated checkpoint, which is `checkpoint_best_ema.pth` instead of
`checkpoint_best_total.pth` (the best of the regular and the EMA model).

## DiffusionDet

`configs/diffdet.densea.res50.yaml` is read by `train_net_densea.py` of densea-diffusiondet, which provides
the early-stopping hook, the best-checkpoint hook and the sparse periodic checkpointer used here. Choices:

- Batch size 8 without gradient accumulation (`--batch 4` in `launch_dd.py` if memory is short; the number of epochs
  is unchanged and the iteration count doubles).
- 300 proposals, focal loss alpha 0.25 / gamma 2 and weight decay 0.05 for every strategy.
- Repeat-factor sampling disabled (`TrainingSampler`, threshold 0). With a threshold of 0.05, Metal_Debris
  (frequency about 0.0065) would be repeated about 2.8 times and Can (about 0.025) about 1.4 times; the other classes
  would stay at 1.
- Training resolution: shortest side 640-800 (maximum 960); VAL and TEST at 800 (maximum 960).
- Evaluation with `SAMPLE_STEP` 1 for selection and for TEST.
- Paths of the COCO files and of the image root are read from `DENSEA_TRAIN_JSON`, `DENSEA_VALID_JSON`,
  `DENSEA_TEST_JSON` and `DENSEA_IMAGE_ROOT`, which `launch_dd.py` sets per experiment.

### Backbone comparison

`configs/diffusiondet/diffdet.densea.{res101,swintiny,swinsmall,swinbase}.yaml` replace only the backbone of the
common R-50 configuration (Real-Only, seed 21). Swin-T and Swin-S weights come from the official ImageNet-1k
checkpoints converted with `run/convert_swin_d2.py` (`--check <config>` verifies that every backbone tensor is
loaded); Swin-B uses the ImageNet-22k weights. Launch with
`python /kit/run/launch_dd.py --exp exp07_real100 --cfg /kit/configs/diffusiondet/<config>.yaml --tag <backbone>`;
the run goes to `/runs/diffusiondet/exp07_real100__<tag>`. The comparison is computed by
`analysis/full_val/dd_backbones_bootstrap.py`.
