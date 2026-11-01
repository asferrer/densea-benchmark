# rtdetrv2, 100% real, 3 seeds: AP on TEST (n = 2207 images, 217 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 44.24 | 65.04 | 47.24 | [40.64; 49.91] |
| exp07_real100_s2 | 22 | 43.62 | 63.91 | 47.40 | [40.17; 48.63] |
| exp07_real100_s3 | 23 | 42.39 | 62.57 | 46.17 | [39.22; 47.90] |
| exp08_real_synth_minor | 21 | 45.29 | 66.23 | 49.79 | [41.71; 50.67] |
| exp08_real_synth_minor_s2 | 22 | 43.23 | 63.92 | 46.90 | [40.16; 47.88] |
| exp08_real_synth_minor_s3 | 23 | 43.49 | 63.97 | 47.36 | [40.36; 48.63] |
| exp06v11_synth100_real100 | 21 | 43.63 | 63.98 | 47.04 | [40.29; 48.69] |
| exp06v11_synth100_real100_s2 | 22 | 43.23 | 63.31 | 46.74 | [40.00; 47.70] |
| exp06v11_synth100_real100_s3 | 23 | 43.08 | 63.10 | 46.32 | [39.67; 48.10] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 43.42 | [40.11; 48.67] | 0.94 | 63.84 | 46.94 | exp07_real100_s2 (43.62) |
| MF clean-bg | 44.00 | [40.77; 49.06] | 1.12 | 64.70 | 48.02 | exp08_real_synth_minor_s3 (43.49) |
| Std copy-paste | 43.31 | [40.03; 48.18] | 0.29 | 63.46 | 46.70 | exp06v11_synth100_real100_s2 (43.23) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | +0.59 | [-0.42; +1.49] | 0.2040 |
| MF clean-bg − Real-Only | AP50 | +0.86 | [-0.58; +2.37] | 0.2180 |
| MF clean-bg − Real-Only | AP75 | +1.08 | [-0.60; +2.44] | 0.1870 |
| Std copy-paste − Real-Only | AP | -0.11 | [-1.60; +1.06] | 0.8030 |
| Std copy-paste − Real-Only | AP50 | -0.38 | [-2.42; +1.15] | 0.5450 |
| Std copy-paste − Real-Only | AP75 | -0.24 | [-2.28; +1.35] | 0.7250 |
| MF clean-bg − Std copy-paste | AP | +0.69 | [-0.21; +1.96] | 0.1410 |
| MF clean-bg − Std copy-paste | AP50 | +1.24 | [-0.11; +3.22] | 0.0750 |
| MF clean-bg − Std copy-paste | AP75 | +1.32 | [-0.02; +2.75] | 0.0550 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | +1.05 | [-0.28; +2.40] | 0.1150 |
| Std copy-paste − Real-Only | 21 | -0.60 | [-2.29; +0.96] | 0.4450 |
| MF clean-bg − Std copy-paste | 21 | +1.66 | [+0.14; +3.24] | 0.0300 |
| MF clean-bg − Real-Only | 22 | -0.39 | [-1.70; +0.91] | 0.6070 |
| Std copy-paste − Real-Only | 22 | -0.39 | [-2.27; +1.40] | 0.6330 |
| MF clean-bg − Std copy-paste | 22 | +0.00 | [-1.27; +1.78] | 0.9440 |
| MF clean-bg − Real-Only | 23 | +1.09 | [-0.61; +2.40] | 0.1530 |
| Std copy-paste − Real-Only | 23 | +0.68 | [-0.86; +1.70] | 0.4910 |
| MF clean-bg − Std copy-paste | 23 | +0.41 | [-0.75; +2.01] | 0.3810 |

The CI of the mean reflects the variation of TEST with the 3 seeds used; the SD across seeds reflects the variation of training.
