# RT-DETRv2 exp08_real_synth_minor_s3 (MF clean-bg, seed 23) on the Jetson, mode rt_final_15W: VAL AP (204 groups of the group partition; group bootstrap, B = 2000)

| Model | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| PyTorch FP32 (PC) | 43.79 | 63.65 | 46.27 | [39.29; 49.38] |
| TensorRT FP16 | 43.62 | 63.62 | 46.18 | [39.11; 49.22] |
| TensorRT INT8 mixed | 43.60 | 63.23 | 46.57 | [39.03; 49.24] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| TensorRT FP16 − PyTorch FP32 (PC) | AP | -0.17 | [-0.34; +0.02] | 0.0700 |
| TensorRT FP16 − PyTorch FP32 (PC) | AP50 | -0.03 | [-0.40; +0.36] | 0.9700 |
| TensorRT FP16 − PyTorch FP32 (PC) | AP75 | -0.09 | [-0.50; +0.48] | 0.6510 |
| TensorRT INT8 mixed − TensorRT FP16 | AP | -0.02 | [-0.41; +0.41] | 0.8700 |
| TensorRT INT8 mixed − TensorRT FP16 | AP50 | -0.39 | [-0.80; +0.23] | 0.1880 |
| TensorRT INT8 mixed − TensorRT FP16 | AP75 | +0.39 | [-0.31; +1.09] | 0.2260 |
| TensorRT INT8 mixed − PyTorch FP32 (PC) | AP | -0.19 | [-0.59; +0.25] | 0.3190 |
| TensorRT INT8 mixed − PyTorch FP32 (PC) | AP50 | -0.41 | [-0.88; +0.27] | 0.2250 |
| TensorRT INT8 mixed − PyTorch FP32 (PC) | AP75 | +0.30 | [-0.38; +1.12] | 0.3920 |
