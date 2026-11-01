# DiffusionDet exp07 (seed 21) on the Jetson, mode dd_15W: VAL AP (204 groups of the group partition; group bootstrap, B = 2000)

| Model | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| PyTorch FP32 | 33.76 | 55.35 | 35.57 | [30.01; 38.56] |
| TensorRT FP32 | 33.63 | 55.21 | 35.50 | [29.86; 38.45] |
| TensorRT mixed (FP32 heads) | 33.71 | 55.14 | 35.51 | [29.91; 38.50] |
| TensorRT FP16 | 29.86 | 51.54 | 31.43 | [26.32; 33.97] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP | -0.05 | [-0.15; +0.01] | 0.0970 |
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP50 | -0.21 | [-0.43; +0.01] | 0.0590 |
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP75 | -0.05 | [-0.17; +0.10] | 0.5160 |
| TensorRT FP32 − PyTorch FP32 | AP | -0.13 | [-0.21; -0.04] | 0.0060 |
| TensorRT FP32 − PyTorch FP32 | AP50 | -0.14 | [-0.25; +0.05] | 0.1790 |
| TensorRT FP32 − PyTorch FP32 | AP75 | -0.07 | [-0.43; +0.13] | 0.6220 |
| TensorRT FP16 − PyTorch FP32 | AP | -3.90 | [-5.64; -3.06] | 0.0000 |
| TensorRT FP16 − PyTorch FP32 | AP50 | -3.82 | [-6.12; -2.55] | 0.0000 |
| TensorRT FP16 − PyTorch FP32 | AP75 | -4.14 | [-6.94; -2.53] | 0.0000 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP | +0.08 | [-0.05; +0.15] | 0.2800 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP50 | -0.07 | [-0.38; +0.16] | 0.5570 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP75 | +0.02 | [-0.21; +0.45] | 0.9770 |
