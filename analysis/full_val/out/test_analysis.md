# Per-class analysis on TEST (n = 2207 images, 217 groups of the group partition; group bootstrap, B = 2000; 3 seeds per arm)

Confidence threshold of the errors: per run, the one that maximises the global (micro) F1 on VAL at IoU 0.5 (greedy matching by score and class), applied unchanged to TEST. The AP values do not depend on the threshold.

## rtdetrv2

Confidence threshold selected on VAL (maximum F1) per run:

| Run | Threshold | F1 on VAL | P on VAL | R on VAL |
|---|---|---|---|---|
| exp07_real100 | 0.663 | 0.750 | 0.847 | 0.672 |
| exp07_real100_s2 | 0.621 | 0.747 | 0.819 | 0.687 |
| exp07_real100_s3 | 0.584 | 0.736 | 0.777 | 0.698 |
| exp08_real_synth_minor | 0.552 | 0.759 | 0.802 | 0.721 |
| exp08_real_synth_minor_s2 | 0.637 | 0.752 | 0.848 | 0.676 |
| exp08_real_synth_minor_s3 | 0.628 | 0.758 | 0.842 | 0.690 |
| exp06v11_synth100_real100 | 0.596 | 0.744 | 0.792 | 0.701 |
| exp06v11_synth100_real100_s2 | 0.647 | 0.745 | 0.823 | 0.681 |
| exp06v11_synth100_real100_s3 | 0.647 | 0.744 | 0.835 | 0.671 |

### rtdetrv2: AP per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 37.36 [23.68; 54.30] | 38.72 [24.73; 55.22] | 38.86 [24.67; 56.15] | +1.36 [-0.62; +3.41] (p = 0.210) | +1.50 [-1.35; +3.85] (p = 0.299) |
| Can | 26.19 [17.68; 38.23] | 28.24 [19.54; 39.35] | 30.16 [20.63; 43.34] | +2.04 [-1.59; +5.19] (p = 0.310) | +3.96 [-0.99; +8.94] (p = 0.123) |
| Fishing_Net | 40.00 [28.52; 55.45] | 42.59 [31.04; 57.66] | 39.96 [29.01; 55.31] | +2.59 [+0.79; +4.12] (p = 0.003) | -0.05 [-2.87; +2.31] (p = 0.965) |
| Glove | 72.65 [62.99; 76.56] | 72.67 [62.84; 76.49] | 72.17 [61.42; 76.51] | +0.02 [-0.87; +1.00] (p = 0.961) | -0.47 [-2.50; +0.33] (p = 0.316) |
| Mask | 68.86 [65.62; 71.94] | 68.67 [65.51; 71.72] | 68.42 [64.07; 71.62] | -0.19 [-1.20; +0.81] (p = 0.707) | -0.45 [-2.31; +0.57] (p = 0.494) |
| Metal_Debris | 11.21 [0.56; 31.61] | 8.69 [0.97; 27.92] | 4.06 [0.05; 16.14] | -2.52 [-7.81; +2.65] (p = 0.487) | -7.15 [-17.37; -0.16] (p = 0.023) |
| Plastic_Debris | 69.41 [61.39; 76.83] | 70.57 [63.17; 77.74] | 70.84 [63.61; 77.98] | +1.16 [-0.66; +3.79] (p = 0.349) | +1.43 [-0.34; +4.21] (p = 0.180) |
| Tire | 21.66 [16.76; 32.94] | 21.89 [17.43; 32.38] | 22.04 [16.81; 33.87] | +0.23 [-1.71; +1.43] (p = 0.845) | +0.38 [-0.60; +2.25] (p = 0.534) |
| Total | 43.42 [40.11; 48.67] | 44.00 [40.77; 49.06] | 43.31 [40.03; 48.18] | +0.59 [-0.42; +1.49] (p = 0.204) | -0.11 [-1.60; +1.06] (p = 0.803) |

### rtdetrv2: AP50 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 61.00 [42.51; 81.13] | 63.08 [43.31; 83.22] | 63.03 [45.40; 82.60] | +2.08 [-0.47; +5.02] (p = 0.114) | +2.03 [-1.54; +4.61] (p = 0.286) |
| Can | 42.81 [29.29; 58.60] | 47.29 [33.96; 62.39] | 45.99 [31.52; 61.76] | +4.48 [-2.46; +11.53] (p = 0.255) | +3.17 [-6.74; +11.13] (p = 0.551) |
| Fishing_Net | 71.04 [57.12; 88.39] | 72.73 [59.50; 89.23] | 72.09 [59.97; 87.77] | +1.69 [-0.09; +3.44] (p = 0.067) | +1.04 [-3.06; +5.12] (p = 0.598) |
| Glove | 92.11 [83.80; 94.83] | 92.35 [84.61; 94.89] | 90.86 [81.70; 93.94] | +0.24 [-0.52; +1.79] (p = 0.546) | -1.25 [-2.95; -0.45] (p = 0.002) |
| Mask | 94.25 [91.60; 96.40] | 94.94 [92.25; 96.75] | 93.80 [90.05; 96.13] | +0.69 [-0.90; +1.73] (p = 0.295) | -0.45 [-2.59; +0.63] (p = 0.498) |
| Metal_Debris | 15.67 [0.75; 40.41] | 11.68 [1.31; 32.42] | 6.00 [0.10; 19.29] | -3.99 [-12.37; +3.77] (p = 0.430) | -9.68 [-22.57; -0.27] (p = 0.012) |
| Plastic_Debris | 87.53 [81.82; 92.60] | 87.62 [81.59; 93.04] | 88.01 [82.28; 93.17] | +0.09 [-1.67; +2.37] (p = 0.982) | +0.48 [-1.09; +2.41] (p = 0.681) |
| Tire | 46.30 [37.40; 63.97] | 47.94 [39.74; 65.04] | 47.94 [38.16; 67.88] | +1.64 [-1.29; +3.72] (p = 0.216) | +1.64 [-0.12; +5.00] (p = 0.071) |
| Total | 63.84 [59.78; 70.24] | 64.70 [60.89; 70.61] | 63.46 [59.66; 69.31] | +0.86 [-0.58; +2.37] (p = 0.218) | -0.38 [-2.42; +1.15] (p = 0.545) |

### rtdetrv2: AP75 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 39.32 [23.25; 61.75] | 41.84 [25.31; 63.55] | 42.30 [24.82; 65.08] | +2.52 [-1.21; +6.11] (p = 0.226) | +2.97 [-0.83; +7.16] (p = 0.152) |
| Can | 28.90 [17.49; 46.04] | 32.52 [20.53; 47.83] | 35.32 [22.64; 53.66] | +3.62 [-3.28; +8.98] (p = 0.300) | +6.42 [+1.14; +11.78] (p = 0.019) |
| Fishing_Net | 37.72 [22.69; 59.00] | 41.99 [27.33; 61.66] | 35.64 [21.03; 56.98] | +4.27 [-0.35; +7.98] (p = 0.074) | -2.07 [-6.34; +1.98] (p = 0.337) |
| Glove | 85.40 [73.82; 89.70] | 85.82 [73.71; 90.10] | 84.74 [71.54; 89.87] | +0.43 [-1.15; +2.25] (p = 0.528) | -0.66 [-3.20; +0.45] (p = 0.321) |
| Mask | 78.87 [74.76; 84.47] | 77.99 [73.47; 83.16] | 77.46 [71.70; 82.73] | -0.88 [-2.75; +0.24] (p = 0.148) | -1.41 [-5.11; +0.56] (p = 0.218) |
| Metal_Debris | 13.00 [0.54; 38.89] | 9.64 [0.94; 30.98] | 4.18 [0.05; 18.19] | -3.36 [-10.83; +2.56] (p = 0.420) | -8.83 [-22.91; -0.22] (p = 0.019) |
| Plastic_Debris | 74.29 [64.07; 83.34] | 76.06 [66.76; 84.57] | 75.84 [67.04; 84.04] | +1.78 [-0.47; +4.92] (p = 0.183) | +1.56 [-0.76; +4.86] (p = 0.320) |
| Tire | 18.00 [12.10; 32.62] | 18.27 [13.08; 31.41] | 18.13 [11.79; 33.38] | +0.27 [-2.49; +2.41] (p = 0.863) | +0.13 [-1.43; +2.58] (p = 0.874) |
| Total | 46.94 [42.81; 53.77] | 48.02 [44.11; 54.04] | 46.70 [42.65; 52.78] | +1.08 [-0.60; +2.44] (p = 0.187) | -0.24 [-2.28; +1.35] (p = 0.725) |

### rtdetrv2: AP by size (mean of 3 seeds ± SD across seeds)

| Arm | APs | APm | APl |
|---|---|---|---|
| Real-Only | 13.44 ± 0.84 | 26.50 ± 0.59 | 49.44 ± 0.64 |
| MF clean-bg | 14.25 ± 0.39 | 27.83 ± 1.92 | 49.55 ± 1.38 |
| Std copy-paste | 11.77 ± 1.72 | 27.16 ± 0.49 | 49.47 ± 0.47 |

### rtdetrv2: per-class errors on TEST (threshold selected on VAL, IoU 0.5; mean of 3 seeds)

FN % = GT without a correct detection (no box + confused with another class); FP % = detections of the class that are not hits (background + confusion).

| Arm | Class | n GT | FN % | of which no box % | of which confused % | FP % | FP background % | FP confusion % |
|---|---|---|---|---|---|---|---|---|
| Real-Only | Bottle | 932 | 45.3 | 44.4 | 0.9 | 31.1 | 29.9 | 1.3 |
| Real-Only | Can | 87 | 59.4 | 54.0 | 5.4 | 44.0 | 31.8 | 12.3 |
| Real-Only | Fishing_Net | 606 | 29.8 | 29.6 | 0.2 | 20.3 | 19.6 | 0.7 |
| Real-Only | Glove | 553 | 10.2 | 9.2 | 1.0 | 7.7 | 5.8 | 1.9 |
| Real-Only | Mask | 672 | 10.4 | 8.8 | 1.5 | 8.0 | 7.0 | 1.0 |
| Real-Only | Metal_Debris | 21 | 87.3 | 66.7 | 20.6 | 31.1 | 0.0 | 31.1 |
| Real-Only | Plastic_Debris | 850 | 17.9 | 16.4 | 1.5 | 9.3 | 7.8 | 1.5 |
| Real-Only | Tire | 812 | 60.8 | 60.4 | 0.4 | 38.9 | 38.9 | 0.0 |
| MF clean-bg | Bottle | 932 | 42.8 | 41.7 | 1.1 | 29.4 | 28.0 | 1.4 |
| MF clean-bg | Can | 87 | 57.1 | 49.4 | 7.7 | 36.1 | 21.7 | 14.4 |
| MF clean-bg | Fishing_Net | 606 | 28.1 | 28.1 | 0.1 | 19.5 | 19.1 | 0.4 |
| MF clean-bg | Glove | 553 | 10.0 | 8.9 | 1.1 | 7.3 | 5.8 | 1.6 |
| MF clean-bg | Mask | 672 | 9.9 | 8.7 | 1.2 | 6.2 | 5.2 | 1.0 |
| MF clean-bg | Metal_Debris | 21 | 93.7 | 65.1 | 28.6 | 30.6 | 19.4 | 11.1 |
| MF clean-bg | Plastic_Debris | 850 | 17.3 | 15.9 | 1.4 | 11.5 | 9.6 | 1.9 |
| MF clean-bg | Tire | 812 | 60.8 | 60.4 | 0.4 | 36.7 | 36.6 | 0.1 |
| Std copy-paste | Bottle | 932 | 42.8 | 41.3 | 1.5 | 28.6 | 27.0 | 1.7 |
| Std copy-paste | Can | 87 | 54.4 | 46.7 | 7.7 | 42.2 | 33.4 | 8.9 |
| Std copy-paste | Fishing_Net | 606 | 31.0 | 30.9 | 0.2 | 20.0 | 19.4 | 0.6 |
| Std copy-paste | Glove | 553 | 11.2 | 10.3 | 0.9 | 8.5 | 6.5 | 2.0 |
| Std copy-paste | Mask | 672 | 10.8 | 9.2 | 1.6 | 6.7 | 5.6 | 1.1 |
| Std copy-paste | Metal_Debris | 21 | 95.2 | 69.8 | 25.4 | 83.3 | 50.0 | 33.3 |
| Std copy-paste | Plastic_Debris | 850 | 17.2 | 15.6 | 1.6 | 12.0 | 9.8 | 2.2 |
| Std copy-paste | Tire | 812 | 62.6 | 62.2 | 0.3 | 34.5 | 34.4 | 0.1 |

### rtdetrv2: most frequent confusions (GT -> predicted), % of the GT of the class, mean of 3 seeds

| Arm | GT -> predicted | % of GT | mean count |
|---|---|---|---|
| Real-Only | Mask -> Glove | 1.5 | 10.3 |
| Real-Only | Plastic_Debris -> Bottle | 0.9 | 7.7 |
| Real-Only | Bottle -> Plastic_Debris | 0.6 | 5.3 |
| Real-Only | Glove -> Mask | 0.8 | 4.3 |
| Real-Only | Plastic_Debris -> Can | 0.4 | 3.3 |
| MF clean-bg | Plastic_Debris -> Bottle | 1.0 | 8.3 |
| MF clean-bg | Mask -> Glove | 1.2 | 8.3 |
| MF clean-bg | Bottle -> Plastic_Debris | 0.5 | 5.0 |
| MF clean-bg | Bottle -> Can | 0.5 | 5.0 |
| MF clean-bg | Glove -> Mask | 0.8 | 4.7 |
| Std copy-paste | Bottle -> Plastic_Debris | 1.2 | 11.3 |
| Std copy-paste | Mask -> Glove | 1.6 | 11.0 |
| Std copy-paste | Plastic_Debris -> Bottle | 1.1 | 9.3 |
| Std copy-paste | Glove -> Mask | 0.7 | 3.7 |
| Std copy-paste | Plastic_Debris -> Mask | 0.4 | 3.3 |

### rtdetrv2: deployed model (exp08_real_synth_minor_s3), threshold 0.628, AP 43.49

Count matrix (rows = GT, columns = predicted; last = background).

| GT \ pred | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire | Background |
|---|---|---|---|---|---|---|---|---|---|
| Bottle | 508 | 3 | 1 | 0 | 0 | 0 | 7 | 0 | 413 |
| Can | 2 | 35 | 0 | 0 | 0 | 0 | 5 | 0 | 45 |
| Fishing_Net | 0 | 0 | 432 | 0 | 0 | 0 | 0 | 0 | 174 |
| Glove | 0 | 0 | 0 | 497 | 5 | 0 | 1 | 0 | 50 |
| Mask | 0 | 0 | 0 | 8 | 613 | 0 | 0 | 0 | 51 |
| Metal_Debris | 1 | 3 | 1 | 0 | 0 | 0 | 3 | 0 | 13 |
| Plastic_Debris | 11 | 1 | 0 | 0 | 2 | 0 | 702 | 0 | 134 |
| Tire | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 289 | 519 |
| Background | 152 | 6 | 92 | 27 | 28 | 0 | 76 | 166 | 0 |

## diffusiondet

Confidence threshold selected on VAL (maximum F1) per run:

| Run | Threshold | F1 on VAL | P on VAL | R on VAL |
|---|---|---|---|---|
| exp07_real100 | 0.407 | 0.698 | 0.752 | 0.652 |
| exp07_real100_s2 | 0.445 | 0.703 | 0.765 | 0.650 |
| exp07_real100_s3 | 0.448 | 0.691 | 0.753 | 0.638 |
| exp08_real_synth_minor | 0.367 | 0.715 | 0.741 | 0.691 |
| exp08_real_synth_minor_s2 | 0.415 | 0.712 | 0.775 | 0.658 |
| exp08_real_synth_minor_s3 | 0.440 | 0.706 | 0.766 | 0.655 |
| exp06v11_synth100_real100 | 0.426 | 0.700 | 0.762 | 0.648 |
| exp06v11_synth100_real100_s2 | 0.460 | 0.694 | 0.779 | 0.625 |
| exp06v11_synth100_real100_s3 | 0.375 | 0.694 | 0.736 | 0.657 |

### diffusiondet: AP per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 21.88 [14.14; 34.17] | 23.05 [15.32; 34.55] | 22.51 [13.74; 34.62] | +1.17 [-3.44; +4.71] (p = 0.608) | +0.63 [-4.23; +4.06] (p = 0.855) |
| Can | 16.36 [10.96; 24.20] | 19.35 [12.11; 29.64] | 22.38 [14.51; 32.40] | +2.99 [-2.20; +9.34] (p = 0.273) | +6.02 [+0.88; +10.36] (p = 0.026) |
| Fishing_Net | 31.10 [20.09; 48.22] | 31.79 [21.55; 48.12] | 31.68 [20.73; 47.36] | +0.69 [-1.82; +2.63] (p = 0.487) | +0.58 [-3.20; +3.32] (p = 0.808) |
| Glove | 62.97 [54.71; 66.80] | 62.91 [53.35; 67.75] | 65.00 [55.78; 69.29] | -0.06 [-2.35; +1.16] (p = 0.883) | +2.03 [+0.17; +3.10] (p = 0.032) |
| Mask | 58.11 [54.91; 61.49] | 58.38 [54.40; 61.40] | 59.76 [56.44; 63.49] | +0.27 [-2.00; +1.78] (p = 0.871) | +1.65 [+0.62; +2.91] (p = 0.003) |
| Metal_Debris | 5.92 [1.37; 16.03] | 14.55 [3.28; 29.79] | 1.34 [0.13; 5.32] | +8.63 [-2.13; +18.43] (p = 0.119) | -4.58 [-13.79; +0.01] (p = 0.058) |
| Plastic_Debris | 61.11 [54.21; 67.70] | 61.93 [54.31; 69.17] | 63.83 [56.40; 71.09] | +0.82 [-0.81; +2.38] (p = 0.274) | +2.72 [+1.22; +4.25] (p = 0.000) |
| Tire | 15.46 [11.16; 24.86] | 16.44 [11.61; 25.73] | 16.77 [12.19; 26.25] | +0.98 [-0.32; +2.40] (p = 0.121) | +1.30 [-0.03; +2.57] (p = 0.060) |
| Total | 34.11 [31.83; 38.08] | 36.05 [33.09; 40.49] | 35.41 [32.92; 39.21] | +1.93 [+0.09; +3.58] (p = 0.036) | +1.29 [-0.37; +2.31] (p = 0.129) |

### diffusiondet: AP50 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 45.46 [31.25; 64.65] | 47.39 [34.50; 64.80] | 43.71 [29.42; 61.03] | +1.93 [-5.46; +8.36] (p = 0.609) | -1.74 [-9.73; +3.90] (p = 0.505) |
| Can | 33.33 [22.16; 47.68] | 37.99 [23.23; 56.54] | 37.68 [23.32; 54.32] | +4.66 [-4.10; +13.87] (p = 0.309) | +4.35 [-4.41; +11.41] (p = 0.419) |
| Fishing_Net | 62.20 [47.27; 81.65] | 63.12 [48.33; 82.16] | 61.08 [46.36; 79.09] | +0.92 [-2.65; +4.68] (p = 0.572) | -1.12 [-7.99; +3.80] (p = 0.585) |
| Glove | 89.13 [79.63; 92.24] | 88.93 [78.90; 92.47] | 88.17 [78.36; 91.45] | -0.20 [-2.29; +0.72] (p = 0.797) | -0.96 [-3.01; +0.49] (p = 0.164) |
| Mask | 91.38 [88.52; 94.42] | 91.77 [87.93; 94.73] | 90.70 [86.92; 93.85] | +0.39 [-1.47; +1.55] (p = 0.684) | -0.68 [-2.60; +0.48] (p = 0.345) |
| Metal_Debris | 13.12 [3.04; 25.97] | 29.40 [7.20; 52.15] | 3.45 [0.17; 12.37] | +16.28 [-1.86; +32.49] (p = 0.073) | -9.67 [-21.28; -0.01] (p = 0.038) |
| Plastic_Debris | 85.19 [78.67; 90.95] | 85.44 [78.47; 91.59] | 85.09 [78.13; 91.36] | +0.25 [-1.28; +1.82] (p = 0.681) | -0.10 [-1.84; +1.65] (p = 0.976) |
| Tire | 38.25 [28.42; 57.72] | 39.98 [30.04; 58.79] | 39.35 [29.80; 56.98] | +1.73 [-1.35; +5.23] (p = 0.244) | +1.09 [-3.57; +4.78] (p = 0.576) |
| Total | 57.26 [53.74; 63.19] | 60.50 [55.58; 67.07] | 56.15 [52.45; 61.57] | +3.25 [+0.33; +6.01] (p = 0.035) | -1.10 [-3.63; +0.64] (p = 0.174) |

### diffusiondet: AP75 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 18.95 [10.02; 35.69] | 19.95 [10.76; 35.63] | 21.47 [11.41; 37.02] | +1.00 [-5.00; +5.54] (p = 0.736) | +2.51 [-4.15; +8.09] (p = 0.489) |
| Can | 13.86 [8.73; 21.77] | 16.83 [8.45; 30.35] | 23.80 [14.48; 35.94] | +2.97 [-7.20; +13.78] (p = 0.529) | +9.94 [+2.45; +17.21] (p = 0.007) |
| Fishing_Net | 27.43 [13.75; 49.46] | 28.42 [15.19; 49.66] | 30.41 [16.01; 52.06] | +0.99 [-2.74; +3.54] (p = 0.491) | +2.98 [-1.73; +7.00] (p = 0.203) |
| Glove | 77.29 [67.01; 82.14] | 77.76 [66.70; 83.13] | 79.39 [69.97; 83.73] | +0.47 [-2.08; +2.00] (p = 0.622) | +2.11 [+0.61; +5.44] (p = 0.009) |
| Mask | 66.29 [60.27; 71.95] | 65.17 [59.41; 69.71] | 67.54 [62.73; 73.93] | -1.12 [-4.36; +1.70] (p = 0.415) | +1.25 [-1.02; +5.19] (p = 0.315) |
| Metal_Debris | 5.36 [0.13; 19.86] | 16.03 [1.71; 40.43] | 0.97 [0.14; 4.49] | +10.67 [-4.14; +27.54] (p = 0.173) | -4.39 [-17.73; +0.90] (p = 0.187) |
| Plastic_Debris | 71.63 [61.73; 80.42] | 71.72 [61.54; 80.85] | 73.02 [63.32; 82.20] | +0.09 [-1.62; +2.09] (p = 0.836) | +1.40 [-0.57; +3.50] (p = 0.170) |
| Tire | 9.68 [6.35; 18.67] | 11.44 [7.44; 20.96] | 11.43 [7.55; 22.28] | +1.76 [+0.02; +4.42] (p = 0.044) | +1.74 [+0.01; +4.54] (p = 0.050) |
| Total | 36.31 [33.37; 41.20] | 38.41 [34.76; 43.85] | 38.50 [35.44; 43.23] | +2.10 [-0.41; +4.95] (p = 0.116) | +2.19 [-0.02; +3.85] (p = 0.054) |

### diffusiondet: AP by size (mean of 3 seeds ± SD across seeds)

| Arm | APs | APm | APl |
|---|---|---|---|
| Real-Only | 8.14 ± 1.21 | 21.44 ± 0.97 | 38.51 ± 1.46 |
| MF clean-bg | 8.23 ± 1.48 | 22.10 ± 0.16 | 41.33 ± 0.80 |
| Std copy-paste | 10.72 ± 1.48 | 22.21 ± 0.68 | 40.74 ± 0.89 |

### diffusiondet: per-class errors on TEST (threshold selected on VAL, IoU 0.5; mean of 3 seeds)

FN % = GT without a correct detection (no box + confused with another class); FP % = detections of the class that are not hits (background + confusion).

| Arm | Class | n GT | FN % | of which no box % | of which confused % | FP % | FP background % | FP confusion % |
|---|---|---|---|---|---|---|---|---|
| Real-Only | Bottle | 932 | 60.5 | 57.1 | 3.4 | 36.5 | 33.9 | 2.6 |
| Real-Only | Can | 87 | 65.1 | 55.6 | 9.6 | 55.7 | 46.1 | 9.7 |
| Real-Only | Fishing_Net | 606 | 33.9 | 33.7 | 0.2 | 32.9 | 30.0 | 3.0 |
| Real-Only | Glove | 553 | 13.6 | 11.4 | 2.2 | 8.3 | 5.9 | 2.4 |
| Real-Only | Mask | 672 | 11.5 | 9.6 | 1.9 | 8.8 | 7.7 | 1.1 |
| Real-Only | Metal_Debris | 21 | 85.7 | 61.9 | 23.8 | 51.8 | 25.6 | 26.2 |
| Real-Only | Plastic_Debris | 850 | 17.6 | 16.7 | 1.0 | 14.8 | 10.3 | 4.5 |
| Real-Only | Tire | 812 | 64.8 | 62.3 | 2.5 | 42.1 | 41.8 | 0.3 |
| MF clean-bg | Bottle | 932 | 56.8 | 53.5 | 3.2 | 33.6 | 30.7 | 2.9 |
| MF clean-bg | Can | 87 | 67.8 | 58.6 | 9.2 | 44.6 | 35.4 | 9.3 |
| MF clean-bg | Fishing_Net | 606 | 33.1 | 32.9 | 0.2 | 31.8 | 29.6 | 2.2 |
| MF clean-bg | Glove | 553 | 13.1 | 11.0 | 2.1 | 9.8 | 7.1 | 2.7 |
| MF clean-bg | Mask | 672 | 11.3 | 9.1 | 2.1 | 9.5 | 8.2 | 1.3 |
| MF clean-bg | Metal_Debris | 21 | 76.2 | 60.3 | 15.9 | 39.0 | 25.1 | 13.8 |
| MF clean-bg | Plastic_Debris | 850 | 18.0 | 17.0 | 0.9 | 14.1 | 10.1 | 4.0 |
| MF clean-bg | Tire | 812 | 62.6 | 60.3 | 2.3 | 43.8 | 43.4 | 0.4 |
| Std copy-paste | Bottle | 932 | 58.7 | 55.4 | 3.4 | 39.8 | 35.9 | 3.9 |
| Std copy-paste | Can | 87 | 56.7 | 40.6 | 16.1 | 61.0 | 53.1 | 7.9 |
| Std copy-paste | Fishing_Net | 606 | 37.1 | 36.6 | 0.4 | 33.0 | 30.4 | 2.6 |
| Std copy-paste | Glove | 553 | 13.0 | 11.4 | 1.6 | 9.3 | 6.8 | 2.5 |
| Std copy-paste | Mask | 672 | 11.5 | 9.6 | 1.8 | 9.0 | 7.7 | 1.2 |
| Std copy-paste | Metal_Debris | 21 | 92.1 | 60.3 | 31.8 | 92.2 | 79.5 | 12.7 |
| Std copy-paste | Plastic_Debris | 850 | 18.6 | 15.9 | 2.7 | 15.5 | 10.9 | 4.6 |
| Std copy-paste | Tire | 812 | 64.2 | 62.6 | 1.7 | 40.8 | 40.4 | 0.4 |

### diffusiondet: most frequent confusions (GT -> predicted), % of the GT of the class, mean of 3 seeds

| Arm | GT -> predicted | % of GT | mean count |
|---|---|---|---|
| Real-Only | Bottle -> Plastic_Debris | 2.6 | 24.0 |
| Real-Only | Tire -> Fishing_Net | 1.7 | 13.7 |
| Real-Only | Mask -> Glove | 1.8 | 12.3 |
| Real-Only | Glove -> Mask | 1.2 | 6.7 |
| Real-Only | Plastic_Debris -> Bottle | 0.7 | 5.7 |
| MF clean-bg | Bottle -> Plastic_Debris | 2.3 | 21.0 |
| MF clean-bg | Mask -> Glove | 2.1 | 14.3 |
| MF clean-bg | Tire -> Fishing_Net | 1.1 | 8.7 |
| MF clean-bg | Tire -> Bottle | 1.1 | 8.7 |
| MF clean-bg | Glove -> Mask | 1.5 | 8.3 |
| Std copy-paste | Bottle -> Plastic_Debris | 2.6 | 24.0 |
| Std copy-paste | Plastic_Debris -> Bottle | 1.8 | 15.3 |
| Std copy-paste | Mask -> Glove | 1.7 | 11.7 |
| Std copy-paste | Tire -> Fishing_Net | 1.4 | 11.0 |
| Std copy-paste | Can -> Bottle | 7.2 | 6.3 |

### diffusiondet: deployed model (exp06v11_synth100_real100_s3), threshold 0.375, AP 34.85

Count matrix (rows = GT, columns = predicted; last = background).

| GT \ pred | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire | Background |
|---|---|---|---|---|---|---|---|---|---|
| Bottle | 385 | 2 | 1 | 0 | 0 | 0 | 21 | 2 | 521 |
| Can | 7 | 35 | 0 | 0 | 0 | 3 | 8 | 0 | 34 |
| Fishing_Net | 1 | 0 | 365 | 0 | 0 | 1 | 1 | 0 | 238 |
| Glove | 0 | 0 | 0 | 488 | 7 | 0 | 2 | 0 | 56 |
| Mask | 0 | 0 | 0 | 12 | 597 | 0 | 0 | 0 | 63 |
| Metal_Debris | 2 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 14 |
| Plastic_Debris | 16 | 4 | 1 | 2 | 2 | 0 | 701 | 0 | 124 |
| Tire | 1 | 0 | 10 | 0 | 0 | 0 | 1 | 332 | 468 |
| Background | 275 | 75 | 209 | 37 | 49 | 28 | 108 | 270 | 0 |

## rfdetr

Confidence threshold selected on VAL (maximum F1) per run:

| Run | Threshold | F1 on VAL | P on VAL | R on VAL |
|---|---|---|---|---|
| exp07_real100 | 0.462 | 0.751 | 0.808 | 0.702 |
| exp07_real100_s2 | 0.513 | 0.747 | 0.839 | 0.673 |
| exp07_real100_s3 | 0.473 | 0.751 | 0.805 | 0.703 |
| exp08_real_synth_minor | 0.495 | 0.739 | 0.830 | 0.666 |
| exp08_real_synth_minor_s2 | 0.467 | 0.741 | 0.814 | 0.680 |
| exp08_real_synth_minor_s3 | 0.451 | 0.743 | 0.796 | 0.697 |
| exp06v11_synth100_real100 | 0.509 | 0.750 | 0.828 | 0.685 |
| exp06v11_synth100_real100_s2 | 0.487 | 0.747 | 0.818 | 0.687 |
| exp06v11_synth100_real100_s3 | 0.485 | 0.745 | 0.818 | 0.684 |

### rfdetr: AP per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 44.28 [28.66; 61.64] | 44.03 [28.50; 59.94] | 43.94 [29.03; 59.91] | -0.25 [-4.02; +1.89] (p = 0.613) | -0.33 [-5.19; +2.63] (p = 0.699) |
| Can | 27.56 [18.25; 39.55] | 26.29 [18.14; 37.09] | 30.17 [19.74; 41.48] | -1.27 [-5.00; +2.24] (p = 0.470) | +2.61 [-4.56; +8.92] (p = 0.496) |
| Fishing_Net | 45.84 [33.93; 60.65] | 45.76 [33.67; 61.23] | 41.13 [30.44; 55.85] | -0.08 [-2.65; +2.24] (p = 0.889) | -4.71 [-7.77; -2.29] (p = 0.000) |
| Glove | 68.55 [58.26; 73.12] | 66.34 [55.32; 71.59] | 68.37 [57.78; 73.05] | -2.21 [-4.83; -0.77] (p = 0.000) | -0.18 [-2.05; +1.25] (p = 0.886) |
| Mask | 59.63 [55.70; 63.48] | 58.89 [55.68; 63.13] | 62.00 [58.83; 66.16] | -0.74 [-2.52; +2.19] (p = 0.498) | +2.37 [+0.23; +6.02] (p = 0.025) |
| Metal_Debris | 12.16 [0.91; 38.16] | 4.75 [0.66; 19.94] | 5.05 [0.40; 16.75] | -7.41 [-21.61; +0.77] (p = 0.177) | -7.11 [-23.26; +4.26] (p = 0.303) |
| Plastic_Debris | 72.67 [65.05; 79.60] | 73.09 [66.07; 79.71] | 73.42 [66.30; 80.14] | +0.41 [-1.27; +2.80] (p = 0.764) | +0.75 [-1.06; +3.06] (p = 0.457) |
| Tire | 25.08 [18.62; 38.56] | 25.71 [19.69; 38.91] | 24.63 [17.97; 38.33] | +0.63 [-0.72; +1.68] (p = 0.352) | -0.45 [-1.74; +0.71] (p = 0.405) |
| Total | 44.47 [40.74; 50.35] | 43.11 [39.88; 48.01] | 43.59 [40.67; 48.13] | -1.36 [-3.21; +0.02] (p = 0.054) | -0.88 [-3.18; +0.88] (p = 0.353) |

### rfdetr: AP50 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 69.86 [53.67; 85.54] | 69.53 [52.17; 85.02] | 69.93 [54.41; 84.91] | -0.34 [-5.20; +2.56] (p = 0.648) | +0.07 [-6.66; +4.15] (p = 0.895) |
| Can | 46.02 [30.63; 62.02] | 46.42 [32.61; 61.48] | 47.90 [30.84; 64.43] | +0.39 [-5.34; +5.86] (p = 0.942) | +1.87 [-7.91; +10.54] (p = 0.789) |
| Fishing_Net | 72.48 [59.43; 87.52] | 70.66 [57.39; 86.51] | 68.00 [54.91; 83.93] | -1.82 [-4.54; +0.95] (p = 0.191) | -4.48 [-9.40; -1.03] (p = 0.008) |
| Glove | 90.90 [81.61; 94.07] | 89.06 [78.15; 93.47] | 89.46 [79.66; 93.03] | -1.84 [-5.71; -0.04] (p = 0.042) | -1.44 [-4.07; +0.18] (p = 0.078) |
| Mask | 89.10 [83.43; 93.13] | 89.17 [84.28; 92.54] | 90.87 [86.42; 93.70] | +0.07 [-2.82; +2.81] (p = 0.980) | +1.77 [-1.48; +5.64] (p = 0.291) |
| Metal_Debris | 15.68 [1.24; 44.41] | 7.54 [1.02; 23.11] | 6.53 [0.68; 19.00] | -8.14 [-25.44; +4.61] (p = 0.308) | -9.15 [-27.17; +3.95] (p = 0.239) |
| Plastic_Debris | 86.97 [80.51; 92.56] | 88.38 [82.50; 93.31] | 87.34 [81.16; 92.74] | +1.41 [-0.21; +3.27] (p = 0.094) | +0.38 [-1.14; +2.04] (p = 0.687) |
| Tire | 51.41 [40.21; 71.73] | 52.81 [42.98; 71.16] | 51.05 [39.26; 71.44] | +1.40 [-1.84; +3.74] (p = 0.516) | -0.36 [-2.79; +1.74] (p = 0.682) |
| Total | 65.30 [60.95; 71.59] | 64.20 [60.30; 69.62] | 63.89 [60.20; 69.04] | -1.11 [-3.44; +0.75] (p = 0.235) | -1.42 [-4.10; +0.66] (p = 0.173) |

### rfdetr: AP75 per class (mean of 3 seeds; 95% CI from the group bootstrap; Δ against Real-Only with CI and p)

| Class | Real-Only | MF clean-bg | Std copy-paste | Δ MF clean-bg | Δ Std copy-paste |
|---|---|---|---|---|---|
| Bottle | 47.14 [26.07; 71.32] | 48.07 [26.20; 71.84] | 47.01 [26.84; 69.51] | +0.93 [-2.97; +3.57] (p = 0.756) | -0.12 [-6.69; +3.80] (p = 0.846) |
| Can | 28.93 [18.59; 43.85] | 26.26 [16.92; 39.75] | 32.47 [20.44; 47.93] | -2.67 [-8.22; +2.38] (p = 0.308) | +3.53 [-4.98; +11.96] (p = 0.452) |
| Fishing_Net | 45.73 [30.61; 65.01] | 45.92 [29.79; 66.49] | 39.01 [24.03; 59.30] | +0.19 [-2.43; +3.45] (p = 0.948) | -6.72 [-12.40; -2.07] (p = 0.008) |
| Glove | 79.83 [68.41; 84.38] | 76.53 [64.31; 82.08] | 79.20 [67.49; 84.11] | -3.30 [-8.49; -0.53] (p = 0.014) | -0.64 [-4.44; +1.72] (p = 0.722) |
| Mask | 64.92 [59.00; 72.69] | 64.69 [59.76; 72.54] | 69.54 [63.46; 77.78] | -0.23 [-2.50; +3.52] (p = 0.773) | +4.62 [+1.29; +9.46] (p = 0.010) |
| Metal_Debris | 13.41 [1.11; 43.00] | 4.71 [0.67; 20.24] | 4.91 [0.31; 17.05] | -8.70 [-25.60; +0.31] (p = 0.096) | -8.50 [-27.30; +3.51] (p = 0.269) |
| Plastic_Debris | 79.41 [71.08; 86.77] | 79.07 [70.99; 86.28] | 79.40 [71.56; 86.67] | -0.35 [-2.40; +2.63] (p = 0.719) | -0.02 [-2.07; +2.55] (p = 0.948) |
| Tire | 22.63 [14.63; 40.38] | 23.34 [15.61; 41.52] | 21.52 [13.57; 38.70] | +0.71 [-0.91; +2.44] (p = 0.326) | -1.11 [-4.26; +0.77] (p = 0.284) |
| Total | 47.75 [43.23; 54.83] | 46.07 [41.90; 52.40] | 46.63 [42.87; 52.64] | -1.68 [-3.97; +0.03] (p = 0.054) | -1.12 [-3.91; +0.99] (p = 0.317) |

### rfdetr: AP by size (mean of 3 seeds ± SD across seeds)

| Arm | APs | APm | APl |
|---|---|---|---|
| Real-Only | 7.26 ± 2.36 | 25.99 ± 0.13 | 50.88 ± 0.04 |
| MF clean-bg | 5.57 ± 0.61 | 25.45 ± 0.14 | 48.92 ± 0.23 |
| Std copy-paste | 6.62 ± 0.49 | 26.40 ± 0.37 | 49.94 ± 0.34 |

### rfdetr: per-class errors on TEST (threshold selected on VAL, IoU 0.5; mean of 3 seeds)

FN % = GT without a correct detection (no box + confused with another class); FP % = detections of the class that are not hits (background + confusion).

| Arm | Class | n GT | FN % | of which no box % | of which confused % | FP % | FP background % | FP confusion % |
|---|---|---|---|---|---|---|---|---|
| Real-Only | Bottle | 932 | 43.9 | 41.6 | 2.3 | 19.5 | 15.7 | 3.9 |
| Real-Only | Can | 87 | 70.9 | 64.0 | 6.9 | 29.4 | 25.2 | 4.3 |
| Real-Only | Fishing_Net | 606 | 39.0 | 39.0 | 0.0 | 17.4 | 15.2 | 2.2 |
| Real-Only | Glove | 553 | 14.2 | 12.9 | 1.3 | 7.4 | 4.7 | 2.7 |
| Real-Only | Mask | 672 | 19.1 | 15.1 | 4.0 | 8.5 | 7.6 | 0.9 |
| Real-Only | Metal_Debris | 21 | 90.5 | 71.4 | 19.1 | 58.9 | 26.1 | 32.8 |
| Real-Only | Plastic_Debris | 850 | 20.8 | 19.4 | 1.4 | 9.0 | 6.1 | 2.9 |
| Real-Only | Tire | 812 | 63.9 | 63.7 | 0.2 | 30.8 | 30.7 | 0.1 |
| MF clean-bg | Bottle | 932 | 46.6 | 44.6 | 2.0 | 15.9 | 11.1 | 4.8 |
| MF clean-bg | Can | 87 | 70.9 | 62.5 | 8.4 | 31.2 | 22.3 | 9.0 |
| MF clean-bg | Fishing_Net | 606 | 40.5 | 40.5 | 0.0 | 20.0 | 17.9 | 2.1 |
| MF clean-bg | Glove | 553 | 17.9 | 15.4 | 2.5 | 9.1 | 5.7 | 3.3 |
| MF clean-bg | Mask | 672 | 22.4 | 18.6 | 3.8 | 9.8 | 7.9 | 1.9 |
| MF clean-bg | Metal_Debris | 21 | 95.2 | 69.8 | 25.4 | 78.3 | 56.7 | 21.7 |
| MF clean-bg | Plastic_Debris | 850 | 19.9 | 17.7 | 2.2 | 9.8 | 7.1 | 2.6 |
| MF clean-bg | Tire | 812 | 65.6 | 65.4 | 0.1 | 27.0 | 27.0 | 0.0 |
| Std copy-paste | Bottle | 932 | 43.1 | 40.1 | 3.0 | 16.9 | 14.0 | 2.9 |
| Std copy-paste | Can | 87 | 57.1 | 47.5 | 9.6 | 35.2 | 25.4 | 9.8 |
| Std copy-paste | Fishing_Net | 606 | 40.6 | 40.4 | 0.2 | 26.3 | 23.7 | 2.6 |
| Std copy-paste | Glove | 553 | 14.5 | 13.1 | 1.4 | 10.4 | 7.8 | 2.6 |
| Std copy-paste | Mask | 672 | 17.6 | 15.1 | 2.5 | 9.1 | 8.1 | 1.0 |
| Std copy-paste | Metal_Debris | 21 | 90.5 | 71.4 | 19.1 | 91.1 | 61.3 | 29.8 |
| Std copy-paste | Plastic_Debris | 850 | 19.4 | 16.8 | 2.6 | 10.6 | 7.7 | 2.9 |
| Std copy-paste | Tire | 812 | 64.0 | 64.0 | 0.0 | 28.5 | 27.9 | 0.6 |

### rfdetr: most frequent confusions (GT -> predicted), % of the GT of the class, mean of 3 seeds

| Arm | GT -> predicted | % of GT | mean count |
|---|---|---|---|
| Real-Only | Mask -> Glove | 2.0 | 13.7 |
| Real-Only | Plastic_Debris -> Bottle | 1.3 | 11.3 |
| Real-Only | Mask -> Bottle | 1.4 | 9.3 |
| Real-Only | Bottle -> Fishing_Net | 1.0 | 9.3 |
| Real-Only | Bottle -> Plastic_Debris | 0.9 | 8.7 |
| MF clean-bg | Plastic_Debris -> Bottle | 2.1 | 17.7 |
| MF clean-bg | Mask -> Glove | 2.4 | 16.0 |
| MF clean-bg | Glove -> Mask | 2.0 | 11.0 |
| MF clean-bg | Bottle -> Fishing_Net | 1.0 | 9.0 |
| MF clean-bg | Bottle -> Plastic_Debris | 0.9 | 8.3 |
| Std copy-paste | Plastic_Debris -> Bottle | 2.0 | 16.7 |
| Std copy-paste | Mask -> Glove | 2.0 | 13.7 |
| Std copy-paste | Bottle -> Fishing_Net | 1.4 | 12.7 |
| Std copy-paste | Bottle -> Plastic_Debris | 1.1 | 10.7 |
| Std copy-paste | Glove -> Mask | 1.0 | 5.3 |

### rfdetr: deployed model (exp07_real100), threshold 0.462, AP 44.55

Count matrix (rows = GT, columns = predicted; last = background).

| GT \ pred | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire | Background |
|---|---|---|---|---|---|---|---|---|---|
| Bottle | 534 | 1 | 10 | 0 | 0 | 1 | 10 | 0 | 376 |
| Can | 3 | 26 | 0 | 0 | 0 | 0 | 3 | 0 | 55 |
| Fishing_Net | 0 | 0 | 385 | 0 | 0 | 0 | 0 | 0 | 221 |
| Glove | 0 | 0 | 0 | 477 | 5 | 0 | 2 | 0 | 69 |
| Mask | 8 | 0 | 0 | 13 | 542 | 0 | 4 | 0 | 105 |
| Metal_Debris | 1 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 15 |
| Plastic_Debris | 12 | 0 | 0 | 0 | 0 | 0 | 675 | 0 | 163 |
| Tire | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 308 | 501 |
| Background | 121 | 7 | 64 | 31 | 55 | 1 | 48 | 147 | 0 |
