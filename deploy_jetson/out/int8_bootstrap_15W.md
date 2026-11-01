# RT-DETRv2 exp07 (seed 21) on the Jetson, mode 15W: VAL AP (204 groups of the group partition; group bootstrap, B = 2000)

| Model | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| ONNX FP32 (PC) | 43.01 | 61.84 | 46.10 | [38.19; 48.29] |
| TensorRT FP16 | 42.94 | 61.81 | 46.02 | [38.19; 48.07] |
| TensorRT INT8 mixed | 42.53 | 61.44 | 45.27 | [37.64; 47.37] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| TensorRT INT8 mixed − TensorRT FP16 | AP | -0.41 | [-1.08; -0.13] | 0.0080 |
| TensorRT INT8 mixed − TensorRT FP16 | AP50 | -0.37 | [-1.12; +0.03] | 0.0680 |
| TensorRT INT8 mixed − TensorRT FP16 | AP75 | -0.75 | [-1.80; -0.26] | 0.0040 |
| TensorRT INT8 mixed − ONNX FP32 (PC) | AP | -0.48 | [-1.35; -0.21] | 0.0020 |
| TensorRT INT8 mixed − ONNX FP32 (PC) | AP50 | -0.40 | [-1.40; -0.03] | 0.0380 |
| TensorRT INT8 mixed − ONNX FP32 (PC) | AP75 | -0.83 | [-2.04; -0.32] | 0.0020 |
| TensorRT FP16 − ONNX FP32 (PC) | AP | -0.07 | [-0.37; +0.03] | 0.1520 |
| TensorRT FP16 − ONNX FP32 (PC) | AP50 | -0.04 | [-0.38; +0.03] | 0.1960 |
| TensorRT FP16 − ONNX FP32 (PC) | AP75 | -0.08 | [-0.43; +0.01] | 0.0740 |
