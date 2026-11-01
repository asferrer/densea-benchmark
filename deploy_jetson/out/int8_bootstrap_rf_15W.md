# RF-DETR exp07 (seed 21) on the Jetson, mode rf_15W: VAL AP (204 groups of the group partition; group bootstrap, B = 2000)

| Model | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| PyTorch FP32 (PC) | 43.54 | 63.68 | 46.93 | [39.45; 49.78] |
| TensorRT FP32 | 43.52 | 63.77 | 46.98 | [39.45; 49.80] |
| TensorRT FP16 | 43.54 | 63.73 | 46.87 | [39.43; 49.82] |
| TensorRT INT8 F (attention) | 43.51 | 63.68 | 46.89 | [39.37; 49.83] |
| TensorRT INT8 G (attention + fc1) | 43.31 | 63.45 | 46.56 | [39.23; 49.16] |
| TensorRT INT8 H (attention + MLP) | 42.89 | 63.09 | 46.50 | [38.85; 48.71] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| TensorRT FP16 − PyTorch FP32 (PC) | AP | +0.00 | [-0.06; +0.10] | 0.7360 |
| TensorRT FP16 − PyTorch FP32 (PC) | AP50 | +0.04 | [-0.03; +0.19] | 0.1480 |
| TensorRT FP16 − PyTorch FP32 (PC) | AP75 | -0.06 | [-0.21; +0.14] | 0.8250 |
| TensorRT FP32 − PyTorch FP32 (PC) | AP | -0.02 | [-0.09; +0.06] | 0.6790 |
| TensorRT FP32 − PyTorch FP32 (PC) | AP50 | +0.09 | [+0.01; +0.16] | 0.0320 |
| TensorRT FP32 − PyTorch FP32 (PC) | AP75 | +0.05 | [-0.09; +0.15] | 0.5820 |
| TensorRT FP16 − TensorRT FP32 | AP | +0.02 | [-0.06; +0.13] | 0.5480 |
| TensorRT FP16 − TensorRT FP32 | AP50 | -0.04 | [-0.11; +0.09] | 0.9120 |
| TensorRT FP16 − TensorRT FP32 | AP75 | -0.12 | [-0.24; +0.11] | 0.5510 |
| TensorRT INT8 F (attention) − TensorRT FP16 | AP | -0.04 | [-0.20; +0.11] | 0.7060 |
| TensorRT INT8 F (attention) − TensorRT FP16 | AP50 | -0.04 | [-0.38; +0.18] | 0.5510 |
| TensorRT INT8 F (attention) − TensorRT FP16 | AP75 | +0.03 | [-0.29; +0.34] | 0.9390 |
| TensorRT INT8 G (attention + fc1) − TensorRT FP16 | AP | -0.24 | [-1.36; +0.12] | 0.2670 |
| TensorRT INT8 G (attention + fc1) − TensorRT FP16 | AP50 | -0.27 | [-1.49; +0.18] | 0.2600 |
| TensorRT INT8 G (attention + fc1) − TensorRT FP16 | AP75 | -0.31 | [-1.69; +0.17] | 0.2130 |
| TensorRT INT8 H (attention + MLP) − TensorRT FP16 | AP | -0.65 | [-1.35; -0.18] | 0.0110 |
| TensorRT INT8 H (attention + MLP) − TensorRT FP16 | AP50 | -0.63 | [-1.36; +0.48] | 0.2140 |
| TensorRT INT8 H (attention + MLP) − TensorRT FP16 | AP75 | -0.37 | [-1.68; +0.29] | 0.2350 |
