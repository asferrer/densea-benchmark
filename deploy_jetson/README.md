# Deployment on the Jetson Orin Nano Super

Deployment of the three trained detectors (RT-DETRv2, RF-DETR, DiffusionDet) on an NVIDIA Jetson Orin Nano Super:
ONNX export on the PC, INT8 calibration set, TensorRT engines, VAL accuracy of the engines with a group bootstrap,
latency / power / energy per power mode, and the latency on a desktop GPU (RTX 5090). Everything is evaluated on
VAL of the group partition (2,185 images, 204 groups); TEST is never copied to the Jetson.

## Stack

- Jetson Orin Nano Super Developer Kit, JetPack 6.2.1 (L4T R36.4.7), TensorRT 10.3.0, CUDA 12.6.
- `trtexec` in `/usr/src/tensorrt/bin`. A Python environment with torch (CUDA), pycuda and pycocotools; the
  `tensorrt` bindings live only in the system Python, hence `PYTHONPATH=/usr/lib/python3.10/dist-packages`.
- Power modes (`/etc/nvpmodel.conf`): 0 = 15 W, 1 = 25 W (default), 2 = MAXN_SUPER, 3 = 7 W. The reported modes are
  7 W, 15 W and MAXN_SUPER. `sudo` without password is required for `jetson_clocks`, `nvpmodel` and `tegrastats`.
- PC side: the three detector containers (`/data`, `/kit`, `/runs` mounted), `onnx`, `onnxsim`, `onnxruntime`,
  and NVIDIA ModelOpt 0.19.0 on Linux or WSL (its C++ extension does not build on Windows).

## Layout and conventions

PC side, under `RUNS_DIR/deploy_jetson` (`deploy_layout.py`; `/runs/deploy_jetson` inside the containers): `models/`
(ONNX files and the engines copied back), `dd/` (DiffusionDet EMA weights), `results/` (logs and prediction files copied
back from the Jetson), `boot/<tag>/` (`<name>_val_preds.json` read by the bootstrap scripts) and `latency_5090/` (raw
JSON of `latency_5090.py`).

Jetson side, one deployment folder `WORK_DIR`, required by every shell script: `data/` (`val.json`, the public VAL
annotations, and the VAL images), `models/`, `results/`, `calib/train_100/` (calibration images for the RF-DETR
quantization), the scripts that run there (`eval_trt_val.py`, `eval_rfdetr_val.py`, `eval_utils.py`, `rtdetrv2_io.py`
and `jetson/*`), `dd/` (`eval_dd_val.py`, `export_onnx_diffusiondet.py`, `eval_utils.py`, `dump_inputs.py`,
`densea_res50_jetson.yaml` and `code/`, the DiffusionDet detectron2 project) and `venv_dd/`, `venv_mo/` (created by
`build_dd_env.sh` and `build_mo_env.sh`). Hosts are written `user@jetson-host`; copy the files with any tool.

## Pipeline

| Step | Where | Script |
|---|---|---|
| 1. ONNX export | PC (CPU, in the detector container) | `export_onnx_rtdetrv2.py`, `export_onnx_rfdetr.py`, `export_onnx_diffusiondet.py` |
| 2. INT8 calibration set | PC | `make_calibration_set.py`, `calibration/calib_100.json` |
| 3. INT8 quantization (Q/DQ) | PC (RT-DETRv2), Jetson (RF-DETR) | `quantize_int8_rtdetrv2.py`, `jetson/quant_rf.py` |
| 4. TensorRT engines | Jetson | `jetson/build_engines.sh`, `rf_build.sh`, `build_dd_engine.sh`, `build_dd_mixed.sh`, `rf_int8_build.sh` |
| 5. VAL accuracy | Jetson (PC for ONNX) | `eval_trt_val.py`, `eval_rfdetr_val.py`, `eval_dd_val.py`, `eval_onnx_val.py` |
| 6. Latency, power, energy | Jetson | `jetson/bench_eval.sh`, `rf_bench.sh`, `dd_bench.sh`, `power.py`, `power_modes.sh`, `*_final_series.sh` |
| 7. Bootstrap, layer profile, desktop GPU | PC / Jetson | `int8_bootstrap.py`, `int8_per_class.py`, `jetson/prof_layers.sh` + `prof_regions.py`, `latency_5090.py` |

Engines are built with `trtexec` and its default options (`--memPoolSize=workspace:2048 --skipInference`) in the power
mode in which they are measured. Every script that reads images uses VAL only.

## INT8 calibration

`python deploy_jetson/make_calibration_set.py [--sizes 100] [--seed 21] [--copy-to DIR]` draws calibration images from
the TRAIN annotations of the group partition (`paths.annotations_path("train")`):

- greedy selection balanced by class and by source: each image scores `sum(1/(n_class+1)) + 1/(n_source+1)` over what
  has been selected so far; ties are broken by a shuffle with seed 21 of the images sorted by id;
- nested sets (the 100-image set is the first part of any larger one);
- it exits if any selected group or file name appears in VAL or TEST;
- `calibration/calib_100.json` is the list used by the paper (100 images, 42 groups: 37 CleanSea, 32 Neural_Ocean,
  31 Ocean_garbage; images per class: Plastic_Debris 45, Bottle 43, Can 37, Glove 36, Mask 26, Tire 22, Metal_Debris 20,
  Fishing_Net 15). It stores image id, group, source and classes; images are resolved through the TRAIN annotations.
  Metal_Debris is present in only 67 TRAIN images, so it stops growing at about 500 calibration images;
- `--copy-to DIR` copies the images (prefixed by their id) to take them to the Jetson.

RT-DETRv2 is quantized on the PC and the engine is built on the Jetson: the Q/DQ ONNX does not depend on the hardware,
and the calibration holds every image in memory (1,000 images of 640 x 640 in float32 are about 4.7 GB, too much for
the 8 GB shared memory of the Orin Nano). Preprocessing is the deployment one (`rtdetrv2_io.preprocess`: cv2,
`INTER_AREA` when shrinking, division by 255), not the torchvision bilinear resize with antialiasing of the PyTorch
VAL; the ONNX AP with this preprocessing equals the PyTorch one (43.01 against 42.94 for exp07).

## RT-DETRv2 on the Jetson

```
# 1. export (CPU, RT-DETRv2 container); writes <name>.onnx and <name>_simplified.onnx
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/app/RT-DETR -w /app/RT-DETR densea_rtdetrv2 \
  python /kit/deploy_jetson/export_onnx_rtdetrv2.py -c /kit/configs/rtdetrv2/generated/exp08_real_synth_minor_s3.yml \
  -r /runs/rtdetrv2/exp08_real_synth_minor_s3/best.pth -o /runs/deploy_jetson/models/rtdetrv2_exp08s3.onnx \
  --static-batch --fuse-layers --simplify --check
# 2. mixed INT8 (Linux/WSL): Q/DQ nodes, the three input convolutions backbone/conv1 kept out of INT8
python deploy_jetson/quantize_int8_rtdetrv2.py --onnx $RUNS_DIR/deploy_jetson/models/rtdetrv2_exp08s3_simplified.onnx \
  --calib_list deploy_jetson/calibration/calib_100.json --calibrate_method entropy --exclude_extra '.*backbone/conv1/.*' \
  --output_onnx $RUNS_DIR/deploy_jetson/models/rtdetrv2_exp08s3_int8mix_stem.onnx
# 3. approximate INT8 AP on the PC (onnxruntime simulates Q/DQ; not the AP of the engine)
python deploy_jetson/eval_onnx_val.py --onnx <model>.onnx --out <preds>.json
# 4. Jetson: copy the ONNX files (verify md5), then
MODEL=rtdetrv2_exp08s3 bash build_engines.sh          # FP16 and mixed INT8 ("stem") engines
bash rt_final_series.sh a                              # 15 W with AP, then MAXN_SUPER; `b` for 7 W
```

The export loads the EMA weights, re-parameterizes the RepVGG blocks, replaces `FrozenBatchNorm2d` by `BatchNorm2d`,
fuses the Conv+BN pairs found among 69 candidate modules and exports opset 17 with `images (1,3,640,640)` and
`orig_target_sizes (1,2) = [width, height]` (int32). Outputs have no batch dimension: `labels [300]`, `boxes [300, 4]`, `scores [300]`. The decoder,
post-processor and heads are always excluded from INT8 (`NODES_TO_EXCLUDE` in `quantize_int8_rtdetrv2.py`): deformable
attention and the sigmoid collapse the scores when quantized. `--calibrate_method` is `entropy` or `minmax` (ModelOpt
`max`). Engine names: `<MODEL>_fp16.engine`, `<MODEL>_int8mix_stem.engine`.

## RF-DETR on the Jetson

```
docker exec -e CUDA_VISIBLE_DEVICES= -w /app/densea-rfdetr densea_rfdetr python /kit/deploy_jetson/export_onnx_rfdetr.py \
  --ckpt /runs/rfdetr/exp07_real100/checkpoint_best_ema.pth --out /runs/deploy_jetson/models/rfdetr_exp07.onnx --check 3
bash rf_build.sh rfdetr_exp07 fp16 --fp16 ; bash rf_build.sh rfdetr_exp07 fp32          # on the Jetson
bash rf_bench.sh 15W rfdetr_exp07_fp16 rfdetr_exp07_fp32                                 # NOAP=1: latency and power only
```

- `onnx` 1.17 and `onnxruntime` 1.20 are installed apart (`pip --target ... --no-deps`, added to `PYTHONPATH`) so the
  training environment is untouched.
- The model is rebuilt from the `args` of the checkpoint, `forward_export` is activated and the model is exported with
  opset 17 and shape 1 x 3 x 560 x 560. Outputs: `dets (1,300,4)` normalized cxcywh and `labels (1,300,9)` logits; the
  post-processing (sigmoid, top-300 over 300 x 9, as `PostProcess`) is `eval_rfdetr_val.postprocess`, outside the
  engine. Preprocessing: PIL bilinear to 560 x 560 and ImageNet normalization.
- `--check 3` verifies: input identical to the library VAL transforms; `forward` against `forward_export` <= 2e-6;
  ONNX against PyTorch <= 3e-4; the first 100 detections have the class of the PC reference and scores within 1.2e-5.
- The head has 9 outputs for 8 classes. About 265 of 655,500 detections fall in class 8, which has no ground truth;
  COCOeval ignores them and `eval_rfdetr_val.py` drops them so that the bootstrap does not reject the file.
- Latency uses `trtexec --loadInputs` with the first VAL image already preprocessed (`models/rf_in_input.bin`).

## DiffusionDet on the Jetson

```
python - <<'EOF'      # EMA weights saved under the key "model" (539 of 549 tensors differ from the raw weights)
import torch
ck = torch.load("/runs/diffusiondet/exp06v11_synth100_real100_s3/model_best.pth", map_location="cpu")
torch.save({"model": ck["ema_state"]}, "/runs/deploy_jetson/dd/diffusiondet_exp06v11s3_ema.pth")
EOF
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/app/densea-diffusiondet -w /app/densea-diffusiondet densea_diffusiondet \
  python /kit/deploy_jetson/export_onnx_diffusiondet.py --weights /runs/deploy_jetson/dd/diffusiondet_exp06v11s3_ema.pth \
  --out /runs/deploy_jetson/models/diffusiondet_exp06v11s3.onnx --check 3
# Jetson: environment, then engines, benchmarks and series
WORK_DIR=... BASE_PYTHON=... bash build_dd_env.sh    # venv_dd: torchvision 0.20.1 (CUDA, sm_87), detectron2 8a9d885
bash dd_final_series.sh a                            # FP32, FP16 and mixed engines; 15 W with AP; MAXN_SUPER; `b`: 7 W
```

- **Exported wrapper (`DDCore`).** Normalization, backbone + FPN and the six heads of the single sampling step.
  Inputs: `image` (RGB 0-255, resized to short side 800 / long side <= 960 and padded with the pixel mean to multiples
  of 32), `images_whwh` and `noise (1, 300, 4)`. Outputs: `logits (1,300,8)` and `boxes (1,300,4)`. Outside the engine:
  sigmoid, top-300 over 300 x 8, per-class NMS at IoU 0.5 (`USE_NMS` is on by default, also with `SAMPLE_STEP 1`),
  rescaling to the original size and removal of empty boxes. `DD_CODE_DIR` and `DD_CONFIG` select the detectron2
  project and the YAML (`configs/diffusiondet/densea_res50.yaml`; on the Jetson `jetson/densea_res50_jetson.yaml`).
- `--check 3` verifies the wrapper against the unwrapped model (same scores and boxes) and the ONNX against the wrapper
  (logits <= 2.9e-3, boxes <= 1.3 px). TensorRT and PyTorch get the same noise (`torch.manual_seed(image id)`).
- **Contiguous input.** TensorRT reads the image pointer as dense NCHW. `prepare()` must return a contiguous array
  (`np.ascontiguousarray`, `.contiguous()` before `.cuda()`); with a transposed view the engine runs without any error
  and returns unrelated outputs.
- **Shapes.** The model is traced with a 640 x 960 dummy image and dynamic `H`, `W`. VAL has four shapes after the
  resize: 800 x 800 (2,021 images), 544 x 960 (82), 640 x 960 (43) and 736 x 960 (39). Engines use the profile
  min 544x800 / opt 800x800 / max 800x960.
- **Latency needs real inputs.** The boxes decide the pyramid level of each RoI, so `trtexec` with random inputs does
  not measure this model. `jetson/dump_inputs.py` writes the first 800 x 800 VAL image and its noise
  (`torch.manual_seed(image id)`) and `dd_bench.sh` passes them with `--loadInputs`.
- **Mixed precision.** The six heads need FP32: `CONSTR=prefer bash build_dd_mixed.sh mixhead "/head/*" <model>` keeps
  `/head/*` in FP32 (`--precisionConstraints=prefer --layerPrecisions`) and the rest in FP16. With `obey` the build
  fails because the `RoiAlign` plugin offers no FP32 format inside the constraint, and `--layerOutputTypes` fails on
  `Cast` layers.

## Accuracy evaluation

All APs are COCO bbox AP, AP50 and AP75 on the 2,185 VAL images. `eval_*_val.py` write the predictions and an
`_ap.json` with the AP and the per-image timing of the Python loop (preprocessing on the CPU, synchronized
inference); the reference latency is always `trtexec`. Copy the predictions to
`RUNS_DIR/deploy_jetson/boot/<tag>/<name>_val_preds.json` (`trt_fp16`, `trt_int8mix_stem`, `rt_torch`, `onnx_fp32`,
`rf_torch`, `rf_trt_fp16`, `rf_trt_fp32`, `rf_trt_qF/qG/qH`, `dd_torch`, `dd_trt_fp32`, `dd_trt_mixhead`, `dd_trt_fp16`)
and run:

```
python deploy_jetson/int8_bootstrap.py --set rtdetrv2_final --tag rt_final_15W      # also rfdetr / rf_15W,
python deploy_jetson/int8_bootstrap.py --set diffusiondet_final --tag dd_final_15W  # diffusiondet / dd_15W, rtdetrv2 / 15W
python deploy_jetson/int8_per_class.py --set rtdetrv2_final --tag rt_final_15W
```

Differences are paired: the same B = 2,000 resamples of the 204 groups (`group_weights` of
`analysis/proxy/px_val_bootstrap.py`, seed 21) are used for every engine, and intervals are 2.5 and 97.5 percentiles.
The PyTorch FP32 references are: PC CPU for RT-DETRv2 and RF-DETR, the wrapper on the Jetson GPU for DiffusionDet.
Selection criterion for INT8 or mixed precision: keep at least 99% of the AP of the FP16/FP32 engine on VAL;
otherwise use FP16.

## Latency, power and energy per mode

`bench_eval.sh MODE_LABEL` (RT-DETRv2), `rf_bench.sh` and `dd_bench.sh` do, per engine and mode:

1. `jetson_clocks` (the original configuration is stored first in `results/jetson_clocks_orig.conf`);
2. `tegrastats --interval 100` running during
   `trtexec --loadEngine=... --warmUp=2000 --duration=30 --useSpinWait` (plus `--loadInputs` for RF-DETR and
   DiffusionDet, `--shapes=image:1x3x800x800` for DiffusionDet);
3. `power.py tegrastats.log trtexec.log`: mean `VDD_IN` (the whole board), `VDD_CPU_GPU_CV`, throughput, and
   energy per inference = `VDD_IN` / throughput. The idle `VDD_IN` is measured for 10 s before.

The reported latency is the mean `GPU Compute Time` of `trtexec`. 15 W and MAXN_SUPER need no reboot (`echo NO | sudo
nvpmodel -m <mode>` declines the reboot prompt); `power_modes.sh` runs both and restores the board. 7 W needs one:
`echo YES | sudo nvpmodel -m 3`, check `nvpmodel -q`, run the benchmark (the `b` part of the series scripts) and switch
back with `echo YES | sudo nvpmodel -m 0`. The AP does not depend on the mode (15 W and MAXN_SUPER give 42.94 FP16 and
42.53 mixed INT8 for RT-DETRv2), so it is measured at 15 W only.

## Results on VAL

### Calibration study (RT-DETRv2 exp07, seed 21; onnxruntime on the CPU)

PyTorch AP of the same weights: 42.94.

| Model | AP | AP50 | AP75 | % of FP32 AP |
|---|---|---|---|---|
| ONNX FP32 (cv2 preprocessing) | 43.01 | 61.83 | 46.10 | 100 |
| INT8 entropy, 100 images | 39.96 | 58.78 | 42.64 | 92.9 |
| INT8 entropy, 500 | 39.77 | 58.88 | 42.30 | 92.5 |
| INT8 entropy, 1,000 | 40.23 | 59.25 | 43.07 | 93.5 |
| INT8 MinMax, 100 | 39.95 | 58.80 | 42.67 | 92.9 |

No variant reaches 99% (42.58): all lose about 3 points, with no clear trend in the number of images or the method, so
the loss does not come from the calibration but from which layers are quantized. Mixed-precision variants
(`--exclude_extra`; entropy, 100 images; all of VAL):

| Variant | INT8 nodes | AP | AP50 | AP75 | % FP32 |
|---|---|---|---|---|---|
| `noaifi` (no encoder attention, `.*encoder/encoder\..*`) | 75 backbone + 35 encoder | 39.93 | 58.82 | 42.64 | 92.9 |
| `noenc` (backbone only, `.*encoder.*`) | 75 backbone | 40.18 | 59.11 | 42.80 | 93.4 |
| **`stem`** (`backbone/conv1`, the 3 convolutions, out) | 72 backbone + 41 encoder | **42.69** | 61.50 | 45.85 | **99.3** |
| `stem1` (only `conv1_1` out) | 74 backbone + 41 encoder | 39.98 | 59.03 | 42.70 | 93.0 |

The loss is in the backbone. Sensitivity sweep on `noenc` over the first 200 VAL images (additionally kept out of INT8):

| Excluded on top of `noenc` | INT8 nodes | AP (200 img.) | % FP32 |
|---|---|---|---|
| (FP32 reference) | 0 | 42.58 | 100 |
| nothing (`noenc`) | 75 | 37.92 | 89.1 |
| `conv1` (`nostem`, the 3 input convolutions) | 72 | 42.18 | 99.1 |

Keeping out instead the residual additions, the pooling, `res_layers.0` or `res_layers.3` leaves 88.6 to 89.7%
(59, 71, 62 and 61 INT8 nodes). Almost all the loss comes from `backbone/conv1`, whose first convolution receives the
image. The configuration used is `stem`: `--exclude_extra '.*backbone/conv1/.*'`, entropy, 100 calibration images.

### RT-DETRv2 exp07 (seed 21) engines, 15 W with `jetson_clocks`

| Engine | AP | AP50 | AP75 | `trtexec` GPU mean (p99) | Python-loop inference | CPU preprocessing |
|---|---|---|---|---|---|---|
| TensorRT FP16 | 42.94 | 61.80 | 46.02 | 44.71 ms (44.82) | 46.4 ms | 3.6 ms |
| TensorRT mixed INT8 `stem` | 42.53 | 61.44 | 45.27 | 34.83 ms (34.92) | 36.6 ms | 3.6 ms |

Paired group bootstrap (`out/int8_bootstrap_15W.md`, B = 2,000), AP differences [95% CI]: mixed INT8 − FP16 −0.41
[−1.08; −0.13]; mixed INT8 − ONNX FP32 (PC) −0.48 [−1.35; −0.21]; FP16 − ONNX FP32 (PC) −0.07 [−0.37; +0.03].

FP16 loses nothing (it gives the PyTorch AP, 42.94). The mixed INT8 engine is 22% faster in GPU latency (1.28x), keeps
99.05% of the FP16 AP (98.9% of FP32), and the loss is small but real (the interval excludes 0; it admits losses up to
about 1.1 points, 2.5%). The PC simulation estimated 42.69 and the engine gives 42.53 (−0.16). Engine build: FP16 15 min
(89 MB), mixed INT8 5 min (54 MB).

### Latency, power and energy per mode (RT-DETRv2 exp07)

Mean `trtexec` GPU time, throughput, mean power of the board (`VDD_IN`) and of `VDD_CPU_GPU_CV`, energy per
inference, idle `VDD_IN`:

| Mode | Engine | GPU mean | Throughput | `VDD_IN` | `VDD_CPU_GPU_CV` | Energy per inference | Idle `VDD_IN` |
|---|---|---|---|---|---|---|---|
| 7 W | FP16 | 103.54 ms | 9.6 FPS | 6.29 W | 1.99 W | 654 mJ | 3.66 W |
| 7 W | mixed INT8 | 80.83 ms | 12.3 FPS | 6.23 W | 1.95 W | 505 mJ | |
| 15 W | FP16 | 44.71 ms | 22.3 FPS | 11.21 W | 5.63 W | 502 mJ | 4.18 W |
| 15 W | mixed INT8 | 34.89 ms | 28.6 FPS | 10.22 W | 4.93 W | 357 mJ | |
| MAXN_SUPER | FP16 | 31.51 ms | 31.7 FPS | 15.54 W | 9.33 W | 490 mJ | 4.74 W |
| MAXN_SUPER | mixed INT8 | 24.51 ms | 40.8 FPS | 14.13 W | 8.22 W | 347 mJ | |

A second pass at 15 W gives 44.67 and 34.80 ms, 497 and 355 mJ. Mixed INT8 is 22% faster in the three modes and uses
29% less energy per image at 15 W and MAXN_SUPER and 23% less at 7 W. MAXN_SUPER lowers latency by 30% with almost the
same energy per image. At 7 W latency is 2.3 times the 15 W one and the board draws about 6.3 W, but energy per image
rises because the constant consumption of the board (3.7 W idle) weighs more at low frequencies: 7 W gives about
12 FPS with mixed INT8 under a 6.3 W power cap, while 15 W and MAXN_SUPER are the most efficient per image.

### Final series of RT-DETRv2 (exp08_real_synth_minor_s3, MF clean-bg, seed 23; all VAL)

Model: the arm with the best mean VAL AP over the 3 seeds of each detector, and in it the seed closest to the mean
(`analysis/full_val/seeds3_bootstrap.py`). Chain as above, with this model. Bootstrap:
`out/int8_bootstrap_rt_final_15W.md`; per class: `out/int8_per_class_rt_final_15W.md`.

| Engine | AP | AP50 | AP75 | − PyTorch [95% CI] | 15 W | MAXN_SUPER | 7 W |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 (PC CPU) | 43.79 | 63.65 | 46.27 | – | – | – | – |
| TensorRT FP16 | 43.62 | 63.62 | 46.18 | −0.17 [−0.34; +0.02] | 44.77 ms · 504 mJ | 31.56 ms · 498 mJ | 103.46 ms · 653 mJ |
| **TensorRT mixed INT8 `stem`** | **43.60** | 63.23 | 46.57 | −0.19 [−0.59; +0.25] | **35.11 ms · 359 mJ** | 24.47 ms · 352 mJ | 81.67 ms · 510 mJ |

- **Mixed INT8 against FP16: −0.02 [−0.41; +0.41]** (99.95% of the FP16 AP). With the exp07 model the loss was −0.41
  [−1.08; −0.13]; with this one it is not distinguishable from zero. AP50: −0.39 [−0.80; +0.23]; AP75: +0.39
  [−0.31; +1.09].
- Latency and energy are those of exp07 (same architecture and engines): 44.77 against 44.71 ms (FP16) and 35.11
  against 34.89 ms (mixed INT8) at 15 W; 653 against 654 mJ (FP16) at 7 W.
- Python loop at 15 W: 46.6 ms of inference with FP16 and 36.9 ms with mixed INT8. Idle `VDD_IN`: 4.14 W at 15 W,
  4.74 W at MAXN_SUPER, 3.66 W at 7 W.

### Final series of DiffusionDet (exp06v11_synth100_real100_s3, Std copy-paste, seed 23; all VAL)

Bootstrap: `out/int8_bootstrap_dd_final_15W.md`; per class: `out/int8_per_class_dd_final_15W.md`. Latency with
`trtexec --loadInputs` on the real 800 x 800 input; energy of the board (`VDD_IN`).

| Engine | AP | AP50 | AP75 | − PyTorch [95% CI] | 15 W | MAXN_SUPER | 7 W |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 (wrapper, Jetson) | 35.46 | 55.69 | 38.70 | – | – | – | – |
| TensorRT FP32 | 35.23 | 55.71 | 38.75 | −0.23 [−0.32; −0.02] | 157.2 ms · 1,674 mJ | 117.3 ms · 1,945 mJ | 392.1 ms · 2,473 mJ |
| **TensorRT mixed (`/head/*` in FP32)** | **35.44** | 55.72 | 38.70 | **−0.02 [−0.11; +0.03]** | **115.2 ms · 1,283 mJ** | 88.4 ms · 1,308 mJ | 267.8 ms · 1,748 mJ |
| TensorRT FP16 | 34.48 | 55.90 | 37.72 | −0.98 [−1.48; −0.44] | 102.4 ms · 1,141 mJ | 77.6 ms · 1,153 mJ | 246.7 ms · 1,585 mJ |

The mixed engine keeps the AP of PyTorch and is 1.36 times faster than FP32 at 15 W. Mixed − FP32: +0.21 [−0.04;
+0.30]. Full FP16 loses less than with exp07 (−0.98 against −3.90) but still loses, with an interval that excludes
zero, mostly in AP75 (−0.98 [−2.17; −0.12]); in AP50 it does not (+0.21 [−0.42; +1.13]). The PyTorch wrapper on the
Jetson GPU takes 359.8 ms of network per image (Python loop, mean).

DiffusionDet exp07 (seed 21), same chain (`out/int8_bootstrap_dd_15W.md`; AP of the evaluation scripts):

| Engine | AP | AP50 | AP75 | 15 W | MAXN_SUPER | 7 W |
|---|---|---|---|---|---|---|
| PyTorch FP32 (wrapper; Python loop 445 ms) | 33.76 | 55.35 | 35.57 | – | – | – |
| TensorRT FP32 | 33.63 | 55.20 | 35.49 | 156.2 ms · 1.85 J | 117.6 ms · 1.93 J | 383.9 ms · 2.50 J |
| **TensorRT mixed (`/head/*` FP32, `prefer`)** | **33.70** | 55.14 | 35.51 | **112.2 ms · 1.24 J** | 86.0 ms · 1.27 J | 266.1 ms · 1.65 J |
| TensorRT FP16 | 29.86 | 51.52 | 31.42 | 101.0 ms · 1.05 J | 76.5 ms · 1.10 J | 244.0 ms · 1.53 J |

AP differences [95% CI]: mixed − PyTorch −0.05 [−0.15; +0.01]; FP32 − PyTorch −0.13 [−0.21; −0.04]; FP16 − PyTorch
−3.90 [−5.64; −3.06]; mixed − FP32 +0.08 [−0.05; +0.15]. The mixed engine is 1.4 times faster than FP32 and about 4
times faster than PyTorch; FP16 gains only 10% in speed and loses 3.9 points. At 7 W the board draws about 6.2 to
6.5 W and each image costs more energy than at 15 W.

### RF-DETR on VAL (exp07, seed 21; `jetson_clocks`, `VDD_IN`)

| Engine | AP | AP50 | AP75 | 7 W | 15 W | MAXN_SUPER |
|---|---|---|---|---|---|---|
| PyTorch FP32 (PC CPU) | 43.55 | 63.70 | 46.96 | – | – | – |
| **TensorRT FP16** | **43.55** | 63.75 | 46.88 | 57.5 ms · 394 mJ | **24.6 ms · 310 mJ** | 17.4 ms · 310 mJ |
| TensorRT FP32 | 43.53 | 63.80 | 46.99 | 142.3 ms · 1.09 J | 61.8 ms · 824 mJ | 48.7 ms · 875 mJ |

Bootstrap (`out/int8_bootstrap_rf_15W.md`): FP16 − PyTorch +0.00 [−0.06; +0.10]; FP32 − PyTorch −0.02 [−0.09; +0.06];
FP16 − FP32 +0.02 [−0.06; +0.13]. Python loop at 15 W: 26.5 ms of inference and 15 ms of PIL preprocessing. At 15 W,
FP16 is faster than the mixed INT8 RT-DETRv2 (24.6 against 34.9 ms), uses less energy per image (310 against 357 mJ)
and has 1 point more AP on VAL (43.55 against 42.53; the exp07 model of both).

## INT8 calibrated RF-DETR

`jetson/rf_int8_build.sh` quantizes on the CPU of the Jetson (`jetson/build_mo_env.sh` creates `venv_mo` with ModelOpt
0.19; `jetson/quant_rf.py`), calibrating with the 100 images of `calibration/calib_100.json` (entropy):

```
quant_rf.py --calib calib/train_100 --n 100 --method entropy --op_types MatMul,Gemm,Conv --disable_mha_qdq \
  --only '.*/attention/(attention/(query|key|value)|output/dense).*,.*/mlp/fc1/.*'          # variant G
bash rf_build.sh rfdetr_exp07_qG int8 --int8 --fp16
```

- Default ModelOpt Q/DQ (whole network or backbone only) gives AP 0.06 and 0.38 on VAL: it also quantizes `Add`, `Mul`,
  `Slice` and `Shape`, and already under onnxruntime the logit correlation with FP32 drops to 0.43-0.65. Only
  `MatMul`/`Gemm`/`Conv` are quantized, with the attention matmuls (Q.K, A.V) in FP16 (variant B).
- Variant B on the first 100 VAL images (`jetson/ap_ort_subset.py`; the absolute AP is lower there): ONNX FP32 with
  onnxruntime 20.77; Q/DQ B with onnxruntime 20.18 (97%); TensorRT FP16 20.75; Q/DQ B with TensorRT 10.3 4.79 (7.70 on
  all VAL); Q/DQ B with TensorRT 10.3 and `--stronglyTyped` 0.84.

The quantization itself works; TensorRT 10.3 breaks when running this Q/DQ graph (a known defect with ModelOpt models:
in NVIDIA forum thread 357835 NVIDIA reproduces it and refers to a future TensorRT release). Located by blocks on the
same 100 images (TensorRT FP16 = 20.75): only the attention projections (44 MatMul) 20.78, only the MLP (22 MatMul)
19.80. Each block works alone; what fails is in the rest of the graph. The variants that TensorRT 10.3 executes
correctly, on all VAL (bootstrap against FP16 in `out/int8_bootstrap_rf_15W.md`):

| Engine | AP | INT8 − FP16 [95% CI] | 15 W | MAXN_SUPER | 7 W |
|---|---|---|---|---|---|
| FP16 | 43.55 | – | 24.58 ms · 306 mJ | 17.43 ms · 308 mJ | 57.43 ms · 393 mJ |
| F: attention | 43.51 | −0.04 [−0.20; +0.11] | 23.89 ms · 287 mJ | 17.68 ms · 297 mJ | 55.03 ms · 386 mJ |
| **G: attention + `fc1`** | **43.31 (99.4%)** | −0.24 [−1.36; +0.12] | **23.64 ms · 275 mJ** | 17.25 ms · 287 mJ | 55.85 ms · 382 mJ |
| H: attention + MLP | 42.90 (98.5%) | −0.65 [−1.35; −0.18] | 22.65 ms · 255 mJ | 16.53 ms · 265 mJ | 53.53 ms · 355 mJ |

H fails the 99% criterion; F and G meet it. G saves the most among them (−10% energy and −4% latency at 15 W, −7%
energy at MAXN_SUPER) but its interval reaches −1.36: in some groups it loses more. F is the safest (no AP loss, 6%
energy saving at 15 W). At 7 W the saving drops to 2-3%. G is the INT8 configuration of RF-DETR, with FP16 as
reference. JetPack 6.2.1 ships only TensorRT 10.3, and G is practically the maximum reachable with it: the full Q/DQ
with random calibration measured 25.17 ms and 271 mJ (slower than G), and NVIDIA reports 36.7 ms in FP16 and 35.4 ms
in INT8 (−3.5%) for a ViT-S on the Orin Nano with its internal TensorRT release.

## Why INT8 speeds up little on the Jetson

`jetson/prof_layers.sh` builds the engines with `--profilingVerbosity=detailed` and runs
`--dumpProfile --separateProfileRun` at 15 W with `jetson_clocks`; `prof_regions.py` splits the time by block and by
precision. The per-layer sum is inflated by the synchronization of each layer (83.6 ms against 43.8 ms measured for
RT-DETRv2 FP16): the splits and the differences are valid, not the totals.

**RT-DETRv2 (mixed INT8, −21% real latency: 43.8 to 34.6 ms in this run).**

| Block | FP16 (profile) | Mixed INT8 (profile) |
|---|---|---|
| Backbone (ResNet) | 16.3 ms | 9.7 ms (INT8) |
| Encoder, convolutional part | 7.2 ms | 3.9 ms (INT8) |
| Decoder: deformable attention (`cross_attn`) | 43.4 ms | 43.5 ms (FP16) |
| Decoder: rest (`enc_output`, queries, heads) | 15.7 ms | 15.7 ms (FP16) |
| FP16 to INT8 reformats | 0.5 ms | 1.1 ms |

INT8 convolutions run about 1.7 times faster (backbone −40%, encoder −46%), not 2 times; the profile saving (−9.9 ms)
matches the real one (−9.3 ms). The decoder is not quantized and is the bottleneck: about 20 ms of the 44 are
deformable attention and the rest of the transformer, Myelin kernels (`Transpose`/`Reshape`/`Slice` of the sampling,
LayerNorm) that are memory bound and also excluded from INT8 for accuracy. Amdahl's law: half of the time is
accelerated 1.7 times.

**RF-DETR (no gain).** 92% of the time is the DINOv2 backbone (windowed ViT-S, 1,600 tokens): attention 34%, MLP 30%,
LayerNorm 16%. Myelin compiles it whole and attention is already fused (14 `_gemm_mha_v2` kernels).

| Engine | GPU (`trtexec`) | Energy per image | Layers in INT8 | Size |
|---|---|---|---|---|
| FP16 | 24.58 ms | 306 mJ | 0% | 58.9 MB |
| `--int8 --fp16` without Q/DQ (implicit) | 24.41 ms | 305 mJ | 0% | 58.9 MB |
| ModelOpt 0.19 Q/DQ, all (random calibration*) | 25.17 ms | 271 mJ | 68% of the time | 32.6 MB |
| Q/DQ, backbone only (random calibration*) | 25.06 ms | 273 mJ | – | 36.2 MB |

\* 16 random tensors: valid for latency and power, **not for the AP** (`quant_rf.py --calib random`).

- Implicit INT8 does nothing: TensorRT warns "Calibrator is not being used. Users must provide dynamic range for all
  tensors" and keeps the network in FP16 (the mode is deprecated since TensorRT 10.5).
- Explicit Q/DQ moves the network to INT8 without gaining speed: MLP −17% (7.7 to 6.4 ms of profile), attention +10%
  (8.8 to 9.7 ms; INT8 MHA kernels are not faster on SM 8.7 at this size), more kernels (`kgen` 105 to 165) and
  reformats (0.1 to 0.7 ms), fewer fusions (`fusion` layers 32 to 10: Q/DQ nodes cut the Myelin fusions), and the
  deformable attention goes from 1.0 to 1.9 ms. Only the power improves: 10.8 W against 12.5 W, 12% less energy per
  image at the same latency.
- Consistent with the literature: ViT-S on Orin Nano with ModelOpt 1-3%, RT-DETR on Thor about 0%, YOLOv8n on Orin
  Nano 1.27x, RF-DETR Base with a proprietary tool 1.26x (1.00x with `--best`).

**DiffusionDet.** The heads do not even admit FP16 (−3.9 AP with exp07), so INT8 could only go to the ResNet backbone
and the FPN: in PyTorch, preprocessing, backbone and FPN take 142 ms (41%) and the 6 heads with 300 boxes and `RoIAlign`
207 ms (59%). If that block ran 1.7 times faster, the Amdahl estimate of the gain would be about 15% (not measured).

## Reference results

`out/` holds the files cited by the paper, written by the scripts: `int8_bootstrap_{15W,rt_final_15W}` (RT-DETRv2 exp07
and final, `--set rtdetrv2` / `rtdetrv2_final`), `int8_bootstrap_rf_15W` (`--set rfdetr`),
`int8_bootstrap_{dd_15W,dd_final_15W}` (`--set diffusiondet` / `diffusiondet_final`), `int8_per_class_{rt_final_15W,
dd_final_15W}` (`int8_per_class.py`) and `latency_5090` (`latency_5090.py --report`; RTX 5090, PyTorch, batch 1).

RTX 5090 (PyTorch FP32, `inference_mode`, batch 1, square VAL images; mean network time, mean end-to-end time and
gross energy per image of the sustained loop; `latency_5090.md` also has FP16 autocast and `no_grad`):

| Detector | Network | End to end | J/img gross (net of idle) |
|---|---|---|---|
| RT-DETRv2 | 18.41 ms | 27.77 ms | 3.87 (2.50) |
| RF-DETR | 13.17 ms | 20.57 ms | 2.61 (1.54) |
| DiffusionDet (1 step) | 27.44 ms | 40.44 ms | 6.53 (4.55) |
