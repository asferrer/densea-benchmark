# DiffusionDet exp06v11_synth100_real100_s3 (Std copy-paste, seed 23) on the Jetson, mode dd_final_15W: VAL AP (204 groups of the group partition; group bootstrap, B = 2000)

| Model | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| PyTorch FP32 | 35.46 | 55.69 | 38.70 | [31.36; 40.24] |
| TensorRT FP32 | 35.23 | 55.71 | 38.75 | [31.22; 40.07] |
| TensorRT mixed (FP32 heads) | 35.44 | 55.72 | 38.70 | [31.33; 40.20] |
| TensorRT FP16 | 34.48 | 55.90 | 37.72 | [30.50; 39.24] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP | -0.02 | [-0.11; +0.03] | 0.2800 |
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP50 | +0.03 | [-0.08; +0.10] | 0.6180 |
| TensorRT mixed (FP32 heads) − PyTorch FP32 | AP75 | -0.01 | [-0.09; +0.09] | 0.8670 |
| TensorRT FP32 − PyTorch FP32 | AP | -0.23 | [-0.32; -0.02] | 0.0240 |
| TensorRT FP32 − PyTorch FP32 | AP50 | +0.02 | [-0.17; +0.18] | 0.8200 |
| TensorRT FP32 − PyTorch FP32 | AP75 | +0.05 | [-0.19; +0.48] | 0.6150 |
| TensorRT FP16 − PyTorch FP32 | AP | -0.98 | [-1.48; -0.44] | 0.0020 |
| TensorRT FP16 − PyTorch FP32 | AP50 | +0.21 | [-0.42; +1.13] | 0.4490 |
| TensorRT FP16 − PyTorch FP32 | AP75 | -0.98 | [-2.17; -0.12] | 0.0270 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP | +0.21 | [-0.04; +0.30] | 0.1100 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP50 | +0.01 | [-0.20; +0.21] | 0.9930 |
| TensorRT mixed (FP32 heads) − TensorRT FP32 | AP75 | -0.06 | [-0.49; +0.20] | 0.6820 |
