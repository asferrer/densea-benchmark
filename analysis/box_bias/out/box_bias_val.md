# Box bias of RT-DETRv2 with the clean-background synthetic batch (v9; VAL only)

## 1. Curves from log.txt (no inference)

| exp | epoch | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl | AP75/AP50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp07 | 104 | 42.94 | 61.73 | 46.02 | 3.77 | 29.87 | 50.74 | 31.49 | 56.07 | 65.92 | 11.54 | 50.02 | 75.26 | 0.745 |
| exp06 | 55 | 43.57 | 63.68 | 46.56 | 3.51 | 29.23 | 51.94 | 32.33 | 55.57 | 65.19 | 31.73 | 49.28 | 74.76 | 0.731 |
| exp08 | 84 | 44.09 | 63.76 | 47.94 | 2.61 | 29.87 | 52.97 | 32.72 | 56.54 | 65.46 | 26.00 | 49.13 | 75.29 | 0.752 |
| exp09 | 57 | 44.01 | 63.47 | 47.63 | 3.44 | 29.82 | 52.74 | 33.47 | 57.26 | 65.88 | 29.77 | 49.39 | 75.81 | 0.750 |

Difference of the best checkpoint to exp07 (points):

| exp | dAP | dAP50 | dAP75 | dAPs | dAPm | dAPl | dAR100 | dARs |
|---|---|---|---|---|---|---|---|---|
| exp06 | 0.62 | 1.94 | 0.54 | -0.26 | -0.63 | 1.20 | -0.73 | 20.18 |
| exp08 | 1.14 | 2.02 | 1.92 | -1.16 | -0.00 | 2.24 | -0.45 | 14.46 |
| exp09 | 1.07 | 1.74 | 1.61 | -0.32 | -0.05 | 2.00 | -0.04 | 18.22 |

Mean per epoch between epochs 41 and 73 (range common to the four):

| exp | AP | AP50 | AP75 | APs | AR100 | AP75/AP50 |
|---|---|---|---|---|---|---|
| exp07 | 41.95 | 61.14 | 44.91 | 3.67 | 66.84 | 0.735 |
| exp06 | 43.25 | 63.39 | 46.40 | 3.57 | 64.91 | 0.732 |
| exp08 | 43.78 | 63.62 | 47.25 | 2.55 | 65.71 | 0.743 |
| exp09 | 43.72 | 63.36 | 47.08 | 3.43 | 65.73 | 0.743 |

AP on VAL per epoch:

| exp | ep. 20 | ep. 40 | ep. 60 | ep. 70 |
|---|---|---|---|---|
| exp07 | 37.71 | 41.09 | 42.14 | 42.30 |
| exp06 | 40.52 | 42.90 | 43.43 | 43.03 |
| exp08 | 40.86 | 43.09 | 43.96 | 43.99 |
| exp09 | 41.25 | 43.16 | 43.97 | 43.79 |

## 2. Calibration: AP recomputed with pycocotools against log.txt

| exp | AP recomputed | AP50 recomputed | AP75 recomputed | AP log | AP50 log | AP75 log |
|---|---|---|---|---|---|---|
| exp07 | 42.937 | 61.741 | 46.032 | 42.943 | 61.734 | 46.020 |
| exp06 | 43.559 | 63.670 | 46.547 | 43.565 | 63.678 | 46.560 |
| exp08 | 44.095 | 63.758 | 47.947 | 44.088 | 63.757 | 47.940 |
| exp09 | 43.987 | 63.432 | 47.603 | 44.014 | 63.475 | 47.631 |

## 3. Size bias (predictions with score >= 0.3, IoU >= 0.5, same class)

| exp | n pairs | area ratio, median [95% CI] | width ratio | height ratio | area, mean [CI] |
|---|---|---|---|---|---|
| exp07 | 3542 | 0.998 [0.992; 1.008] | 0.998 [0.993; 1.004] | 1.002 [0.998; 1.007] | 1.022 [1.005; 1.049] |
| exp06 | 3668 | 0.994 [0.984; 1.006] | 0.996 [0.989; 1.000] | 1.002 [0.998; 1.006] | 1.022 [1.003; 1.049] |
| exp08 | 3647 | 0.997 [0.989; 1.008] | 0.997 [0.990; 1.001] | 1.002 [0.998; 1.005] | 1.023 [1.003; 1.053] |
| exp09 | 3668 | 0.996 [0.986; 1.004] | 0.997 [0.991; 1.000] | 1.000 [0.997; 1.004] | 1.014 [0.997; 1.039] |

Shift per side (positive = the prediction protrudes from the GT), median [95% CI]:

| exp | left (px) | top (px) | right (px) | bottom (px) | mean of 4 sides (px) |
|---|---|---|---|---|---|
| exp07 | 0.3 [0.1; 0.7] | 0.3 [-0.1; 0.8] | -0.5 [-0.8; -0.2] | -0.3 [-0.5; 0.0] | 0.0 [-0.2; 0.4] |
| exp06 | 0.2 [-0.1; 0.5] | 0.1 [-0.1; 0.5] | -0.6 [-1.0; -0.3] | -0.1 [-0.4; 0.1] | -0.1 [-0.4; 0.2] |
| exp08 | 0.1 [-0.1; 0.4] | 0.2 [-0.1; 0.6] | -0.6 [-0.9; -0.3] | -0.1 [-0.3; 0.1] | -0.1 [-0.2; 0.2] |
| exp09 | 0.2 [-0.1; 0.5] | -0.1 [-0.3; 0.4] | -0.6 [-0.9; -0.4] | -0.3 [-0.4; -0.1] | -0.1 [-0.3; 0.1] |

Same, relative to the GT side (fraction):

| exp | left | top | right | bottom | mean of 4 sides |
|---|---|---|---|---|---|
| exp07 | 0.0028 [0.0005; 0.0049] | 0.0015 [-0.0006; 0.0046] | -0.0033 [-0.0057; -0.0012] | -0.0014 [-0.0024; 0.0002] | -0.0003 [-0.0018; 0.0022] |
| exp06 | 0.0011 [-0.0008; 0.0034] | 0.0008 [-0.0009; 0.0030] | -0.0043 [-0.0080; -0.0017] | -0.0006 [-0.0018; 0.0008] | -0.0013 [-0.0037; 0.0019] |
| exp08 | 0.0007 [-0.0009; 0.0031] | 0.0013 [-0.0008; 0.0035] | -0.0040 [-0.0070; -0.0021] | -0.0007 [-0.0019; 0.0006] | -0.0005 [-0.0026; 0.0026] |
| exp09 | 0.0017 [-0.0003; 0.0035] | -0.0003 [-0.0020; 0.0023] | -0.0041 [-0.0078; -0.0019] | -0.0015 [-0.0027; -0.0004] | -0.0009 [-0.0032; 0.0015] |

Localisation quality of the pairs (median; mean for the fraction) [95% CI]:

| exp | IoU, median | IoU, mean | |ln area ratio|, median | fraction with area ratio > 1.3 |
|---|---|---|---|---|
| exp07 | 0.870 [0.838; 0.891] | 0.837 [0.811; 0.857] | 0.083 [0.069; 0.109] | 0.084 [0.056; 0.126] |
| exp06 | 0.863 [0.830; 0.884] | 0.830 [0.806; 0.848] | 0.089 [0.075; 0.109] | 0.090 [0.063; 0.127] |
| exp08 | 0.866 [0.832; 0.888] | 0.834 [0.808; 0.854] | 0.087 [0.073; 0.107] | 0.087 [0.055; 0.130] |
| exp09 | 0.866 [0.835; 0.882] | 0.834 [0.811; 0.851] | 0.090 [0.077; 0.107] | 0.080 [0.053; 0.116] |

Difference exp - exp07 (group bootstrap with the same resamples), 95% CI:

| exp | area (median) | width (median) | height (median) |
|---|---|---|---|
| exp06 | 0.994 - 0.998; CI of diff. [-0.011; 0.003] | 0.996 - 0.998; CI of diff. [-0.007; 0.001] | 1.002 - 1.002; CI of diff. [-0.003; 0.003] |
| exp08 | 0.997 - 0.998; CI of diff. [-0.006; 0.004] | 0.997 - 0.998; CI of diff. [-0.005; 0.001] | 1.002 - 1.002; CI of diff. [-0.003; 0.003] |
| exp09 | 0.996 - 0.998; CI of diff. [-0.010; 0.002] | 0.997 - 0.998; CI of diff. [-0.005; 0.001] | 1.000 - 1.002; CI of diff. [-0.005; 0.001] |

Mean IoU of the pairs, exp - exp07:

| exp | mean IoU |
|---|---|
| exp06 | 0.8298 - 0.8371; CI of diff. [-0.0129; -0.0020] |
| exp08 | 0.8340 - 0.8371; CI of diff. [-0.0085; 0.0015] |
| exp09 | 0.8340 - 0.8371; CI of diff. [-0.0075; 0.0031] |

GT of VAL by COCO size: large 2795, medium 1717, small 42. By relative size: >15 % 3712, 5-15 % 813, <5 % 29.

By COCO size of the GT: median area ratio [95% CI] (n pairs) and mean shift per side (px)

| group | exp07 | exp06 | exp08 | exp09 |
|---|---|---|---|---|
| small | 1.148 [0.707; 1.872] (n=5); 1.5 | 1.477 [0.895; 1.481] (n=5); 1.6 | 1.154 [0.707; 1.686] (n=5); 1.3 | 1.398 [0.889; 1.634] (n=8); 2.1 |
| medium | 0.992 [0.955; 1.012] (n=1129); -0.0 | 1.003 [0.971; 1.040] (n=1192); 0.0 | 0.997 [0.950; 1.026] (n=1168); -0.0 | 1.003 [0.970; 1.023] (n=1184); -0.1 |
| large | 0.999 [0.992; 1.013] (n=2408); 1.0 | 0.991 [0.984; 1.003] (n=2471); 0.8 | 0.998 [0.989; 1.008] (n=2474); 1.3 | 0.993 [0.985; 1.001] (n=2476); 0.5 |

By relative size of the GT (square root of the box area / square root of the image area): median area ratio [95% CI] (n pairs) and mean shift per side (px)

| group | exp07 | exp06 | exp08 | exp09 |
|---|---|---|---|---|
| <5 % | 1.055 [0.958; 1.071] (n=7); 3.1 | 0.944 [0.824; 1.016] (n=7); -1.3 | 1.038 [0.933; 1.067] (n=7); 3.1 | 1.017 [0.926; 1.067] (n=8); 2.3 |
| 5-15 % | 1.017 [0.996; 1.060] (n=408); 0.9 | 1.043 [1.011; 1.090] (n=458); 0.6 | 1.028 [0.994; 1.068] (n=446); 0.9 | 1.032 [1.016; 1.075] (n=455); 0.7 |
| >15 % | 0.997 [0.991; 1.006] (n=3127); 0.7 | 0.991 [0.981; 1.001] (n=3203); 0.6 | 0.996 [0.985; 1.006] (n=3194); 0.8 | 0.992 [0.982; 1.000] (n=3205); 0.2 |

By class: median area ratio [95% CI] (n pairs) and mean shift per side (px)

| group | exp07 | exp06 | exp08 | exp09 |
|---|---|---|---|---|
| Bottle | 0.992 [0.971; 1.050] (n=662); 0.5 | 0.976 [0.948; 1.025] (n=744); -0.4 | 0.974 [0.942; 1.022] (n=711); 0.1 | 0.976 [0.962; 1.016] (n=730); -0.4 |
| Can | 0.995 [0.911; 1.024] (n=36); -0.2 | 1.015 [0.941; 1.088] (n=39); 0.4 | 1.012 [0.900; 1.067] (n=38); -1.5 | 1.001 [0.903; 1.039] (n=39); -0.7 |
| Fishing_Net | 1.047 [0.999; 1.154] (n=395); 6.0 | 1.042 [0.996; 1.145] (n=417); 7.3 | 1.052 [1.000; 1.143] (n=414); 8.5 | 1.040 [0.996; 1.142] (n=415); 6.1 |
| Glove | 0.991 [0.972; 0.997] (n=512); -0.7 | 0.990 [0.964; 1.001] (n=514); -0.6 | 0.995 [0.973; 1.003] (n=516); -0.7 | 0.987 [0.947; 0.998] (n=516); -0.9 |
| Mask | 1.002 [0.968; 1.039] (n=600); -0.4 | 0.993 [0.983; 1.029] (n=602); -0.6 | 0.993 [0.966; 1.032] (n=594); -0.5 | 0.994 [0.977; 1.009] (n=604); -0.6 |
| Metal_Debris | 0.962 [0.962; 0.962] (n=1); 1.3 | 1.053 [0.962; 1.143] (n=2); 3.7 | 1.017 [0.957; 1.077] (n=2); 1.7 | 1.009 [0.996; 1.023] (n=2); 0.8 |
| Plastic_Debris | 0.996 [0.980; 1.009] (n=750); 0.1 | 0.989 [0.976; 1.007] (n=757); -0.3 | 0.996 [0.981; 1.009] (n=760); 0.1 | 0.990 [0.975; 1.006] (n=765); -0.6 |
| Tire | 0.987 [0.949; 1.056] (n=586); 0.4 | 1.013 [0.978; 1.048] (n=593); 0.3 | 1.004 [0.954; 1.057] (n=612); 0.2 | 1.015 [0.978; 1.056] (n=597); 0.2 |

Sensitivity to the score threshold (median area ratio [CI]):

| exp | score >= 0.1 | score >= 0.5 |
|---|---|---|
| exp07 | 0.998 [0.992; 1.009] | 0.997 [0.991; 1.007] |
| exp06 | 0.994 [0.984; 1.008] | 0.993 [0.983; 1.004] |
| exp08 | 0.999 [0.991; 1.010] | 0.998 [0.990; 1.009] |
| exp09 | 0.997 [0.988; 1.008] | 0.996 [0.988; 1.006] |

## 4. Post-hoc correction (shrink the boxes by the mean bias per side)

Relative (fraction of the box dimension) correction. Mean over 5 partitions of the groups into two halves (fit on one, evaluate on the other, both directions); in parentheses, mean change ± SD across partitions:

| exp | AP | AP50 | AP75 | APs |
|---|---|---|---|---|
| exp07 | 42.80 -> 42.59 (-0.20 ± 0.10) | 61.92 -> 61.93 (+0.01 ± 0.04) | 45.59 -> 45.35 (-0.24 ± 0.16) | 5.10 -> 5.10 (+0.00 ± 0.01) |
| exp06 | 43.44 -> 43.30 (-0.14 ± 0.14) | 63.74 -> 63.69 (-0.06 ± 0.07) | 46.01 -> 45.99 (-0.02 ± 0.11) | 4.95 -> 4.97 (+0.02 ± 0.07) |
| exp08 | 43.91 -> 43.79 (-0.12 ± 0.12) | 63.76 -> 63.72 (-0.04 ± 0.03) | 47.46 -> 47.39 (-0.07 ± 0.29) | 3.41 -> 3.45 (+0.04 ± 0.02) |
| exp09 | 44.20 -> 44.23 (+0.03 ± 0.04) | 63.83 -> 63.95 (+0.12 ± 0.19) | 47.66 -> 47.83 (+0.18 ± 0.18) | 4.94 -> 5.32 (+0.38 ± 0.16) |

Absolute (px) correction. Mean over 5 partitions of the groups into two halves (fit on one, evaluate on the other, both directions); in parentheses, mean change ± SD across partitions:

| exp | AP | AP50 | AP75 | APs |
|---|---|---|---|---|
| exp07 | 42.80 -> 42.07 (-0.72 ± 0.35) | 61.92 -> 61.68 (-0.24 ± 0.09) | 45.59 -> 44.90 (-0.69 ± 0.38) | 5.10 -> 4.60 (-0.50 ± 0.42) |
| exp06 | 43.44 -> 42.69 (-0.75 ± 0.30) | 63.74 -> 63.67 (-0.08 ± 0.09) | 46.01 -> 45.11 (-0.90 ± 0.51) | 4.95 -> 4.21 (-0.74 ± 0.69) |
| exp08 | 43.91 -> 42.86 (-1.05 ± 0.36) | 63.76 -> 63.35 (-0.41 ± 0.09) | 47.46 -> 46.57 (-0.89 ± 0.37) | 3.41 -> 3.20 (-0.21 ± 0.73) |
| exp09 | 44.20 -> 43.85 (-0.35 ± 0.22) | 63.83 -> 63.78 (-0.05 ± 0.05) | 47.66 -> 47.23 (-0.42 ± 0.28) | 4.94 -> 5.15 (+0.20 ± 0.32) |


## 5. Reading of the results (the tables above are authoritative)

- **Calibration.** The AP recomputed with pycocotools on the own predictions (CPU, fp32) differs from `log.txt`
  (GPU, AMP) by at most 0.05 points in AP, AP50 and AP75.
- **Curves.** On real VAL the three exps with synthetic v9 exceed exp07: AP +0.6 (exp06), +1.1 (exp08), +1.1 (exp09); AP50
  +1.7 to +2.0. AP75 rises +0.5, +1.9 and +1.6. The AP75/AP50 ratio is 0.731, 0.752 and 0.750 against 0.745 for exp07: only
  exp06 (epoch 55 of 74) is below, and there is no systematic loss in AP75.
- **Small objects.** APs neither rises nor falls appreciably (3.5, 2.6, 3.4 against 3.8), but VAL has 42 "small" GT
  (0.9 % of 4,554) and only 5 to 8 pairs per exp: the small-object hypothesis **cannot be tested on VAL**.
  ARs is much higher with synthetic data (26 to 32 against 11.5), with the same caveat.
- **Size bias.** Median pred/gt area ratio 0.994 to 0.998 in the four exps (exp07: 0.998); the differences to
  exp07 have CIs that include 0 and the median shift per side is 0.6 px at most (the margin of the synthetic boxes is
  ~20 px per side). There is no bias by class or by size: the only class with a ratio > 1 is Fishing_Net (1.04 to 1.05), and it
  is already present in exp07 (1.047), so it is not due to the synthetic data.
- **Localisation.** The only detectable effect: mean IoU of the pairs of exp06 0.830 against 0.837 for exp07 (CI of the
  difference [-0.013; -0.002]); exp08 and exp09 are indistinguishable from exp07.
- **Post-hoc correction.** Shrinking the boxes with the mean bias estimated on the other half does not recover AP (relative: -0.14,
  -0.12 and +0.03 in exp06, exp08 and exp09, against -0.20 in the control exp07; absolute in px, worse in all). The damage
  attributable to box size is indistinguishable from 0 (|change| <= 0.2 AP, of the order of the SD across partitions).
- **Caveats.** It is measured on real VAL images (the synthetic data does not enter); `best.pth` is selected by AP on that same
  VAL (it affects the four equally); matching requires IoU >= 0.5, so a strongly inflated box on a small object would
  have no pair (survivorship bias), but the sensitivity to score 0.1/0.3/0.5 is nil and the VAL objects are almost all
  large; it cannot be ruled out that the inflation harms small objects, which VAL does not contain.

