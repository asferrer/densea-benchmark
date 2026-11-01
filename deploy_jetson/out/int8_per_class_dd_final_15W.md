# DiffusionDet exp06v11_synth100_real100_s3 (Std copy-paste, seed 23), mode dd_final_15W: per-class AP (COCO) on VAL (204 groups of the group partition; group bootstrap, B = 2000)

| Class | Instances | PyTorch FP32 | TensorRT FP32 | TensorRT mixed (FP32 heads) | TensorRT FP16 |
|---|---|---|---|---|---|
| Bottle | 946 | 32.93 | 32.69 | 32.97 | 32.43 |
| Can | 88 | 8.79 | 9.02 | 8.78 | 8.84 |
| Fishing_Net | 614 | 28.00 | 27.83 | 28.16 | 27.40 |
| Glove | 550 | 68.30 | 67.48 | 68.03 | 65.49 |
| Mask | 673 | 57.28 | 56.91 | 57.16 | 55.12 |
| Metal_Debris | 23 | 7.19 | 7.19 | 7.19 | 7.34 |
| Plastic_Debris | 851 | 61.55 | 61.16 | 61.67 | 58.74 |
| Tire | 809 | 19.67 | 19.58 | 19.59 | 20.52 |

## TensorRT mixed (FP32 heads) − PyTorch FP32, per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | +0.04 | [-0.09; +0.16] |
| Can | -0.01 | [-0.44; +0.13] |
| Fishing_Net | +0.16 | [+0.02; +0.38] |
| Glove | -0.27 | [-0.89; -0.14] |
| Mask | -0.12 | [-0.23; +0.20] |
| Metal_Debris | -0.00 | [-0.01; +0.01] |
| Plastic_Debris | +0.12 | [-0.01; +0.26] |
| Tire | -0.09 | [-0.23; +0.01] |

## TensorRT FP32 − PyTorch FP32, per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | -0.24 | [-0.64; +0.28] |
| Can | +0.23 | [-0.02; +0.70] |
| Fishing_Net | -0.17 | [-0.37; +0.10] |
| Glove | -0.82 | [-1.09; +0.35] |
| Mask | -0.37 | [-0.95; -0.21] |
| Metal_Debris | +0.00 | [-0.01; +0.01] |
| Plastic_Debris | -0.40 | [-0.74; -0.02] |
| Tire | -0.09 | [-0.34; +0.31] |

## TensorRT FP16 − PyTorch FP32, per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | -0.50 | [-2.27; +1.14] |
| Can | +0.05 | [-0.98; +1.08] |
| Fishing_Net | -0.59 | [-1.88; +0.62] |
| Glove | -2.81 | [-6.35; -1.38] |
| Mask | -2.16 | [-2.71; +0.28] |
| Metal_Debris | +0.15 | [-0.01; +1.37] |
| Plastic_Debris | -2.82 | [-5.48; -0.45] |
| Tire | +0.84 | [-0.35; +1.75] |

## TensorRT mixed (FP32 heads) − TensorRT FP32, per class

| Class | Difference | 95% CI |
|---|---|---|
| Bottle | +0.28 | [-0.29; +0.72] |
| Can | -0.25 | [-1.02; +0.07] |
| Fishing_Net | +0.33 | [+0.02; +0.62] |
| Glove | +0.56 | [-1.06; +0.85] |
| Mask | +0.25 | [+0.11; +0.91] |
| Metal_Debris | -0.00 | [-0.01; +0.01] |
| Plastic_Debris | +0.52 | [+0.14; +0.85] |
| Tire | +0.00 | [-0.45; +0.28] |
