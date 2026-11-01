# RT-DETRv2 exp08_real_synth_minor_s3 (MF clean-bg, seed 23), mode rt_final_15W: per-class AP (COCO) on VAL (204 groups of the group partition; group bootstrap, B = 2000)

| Class | Instances | PyTorch FP32 (PC) | TensorRT FP16 | TensorRT INT8 mixed |
|---|---|---|---|---|
| Bottle | 946 | 47.27 | 47.22 | 46.67 |
| Can | 88 | 22.46 | 21.36 | 21.26 |
| Fishing_Net | 614 | 34.16 | 34.31 | 35.24 |
| Glove | 550 | 75.72 | 75.97 | 75.79 |
| Mask | 673 | 66.26 | 65.87 | 65.94 |
| Metal_Debris | 23 | 8.66 | 8.69 | 8.83 |
| Plastic_Debris | 851 | 70.11 | 70.05 | 69.59 |
| Tire | 809 | 25.69 | 25.52 | 25.51 |

## TensorRT FP16 − PyTorch FP32 (PC), per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | -0.05 | [-0.32; +0.10] |
| Can | -1.10 | [-2.34; +0.10] |
| Fishing_Net | +0.15 | [-0.11; +0.35] |
| Glove | +0.25 | [-0.42; +0.36] |
| Mask | -0.39 | [-0.50; +0.30] |
| Metal_Debris | +0.03 | [-0.00; +0.35] |
| Plastic_Debris | -0.06 | [-0.24; +0.18] |
| Tire | -0.17 | [-0.46; +0.08] |

## TensorRT INT8 mixed − TensorRT FP16, per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | -0.55 | [-1.27; +0.43] |
| Can | -0.10 | [-1.15; +0.98] |
| Fishing_Net | +0.93 | [-0.12; +1.91] |
| Glove | -0.18 | [-1.35; +0.40] |
| Mask | +0.08 | [-0.73; +0.30] |
| Metal_Debris | +0.14 | [-2.29; +3.70] |
| Plastic_Debris | -0.46 | [-1.04; -0.04] |
| Tire | -0.00 | [-0.54; +0.99] |

## TensorRT INT8 mixed − PyTorch FP32 (PC), per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | -0.60 | [-1.41; +0.34] |
| Can | -1.20 | [-2.39; +0.15] |
| Fishing_Net | +1.08 | [-0.02; +2.08] |
| Glove | +0.07 | [-1.49; +0.39] |
| Mask | -0.31 | [-0.79; +0.27] |
| Metal_Debris | +0.17 | [-2.28; +4.16] |
| Plastic_Debris | -0.52 | [-1.08; -0.04] |
| Tire | -0.17 | [-0.68; +0.76] |
