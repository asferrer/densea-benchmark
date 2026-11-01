# rfdetr, 100% real, 3 seeds: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 43.54 | 63.68 | 46.93 | [39.45; 49.78] |
| exp07_real100_s2 | 22 | 43.37 | 63.65 | 46.33 | [39.23; 49.50] |
| exp07_real100_s3 | 23 | 43.67 | 63.39 | 47.00 | [39.63; 49.99] |
| exp08_real_synth_minor | 21 | 42.95 | 63.47 | 45.32 | [39.01; 48.96] |
| exp08_real_synth_minor_s2 | 22 | 43.00 | 63.55 | 45.37 | [38.94; 48.98] |
| exp08_real_synth_minor_s3 | 23 | 42.93 | 63.28 | 45.63 | [39.05; 48.63] |
| exp06v11_synth100_real100 | 21 | 43.73 | 62.84 | 47.33 | [39.55; 48.43] |
| exp06v11_synth100_real100_s2 | 22 | 43.26 | 62.12 | 46.52 | [39.20; 48.30] |
| exp06v11_synth100_real100_s3 | 23 | 43.09 | 61.94 | 46.68 | [38.93; 47.76] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 43.52 | [39.44; 49.76] | 0.15 | 63.57 | 46.75 | exp07_real100 (43.54) |
| MF clean-bg | 42.96 | [38.97; 48.74] | 0.04 | 63.43 | 45.44 | exp08_real_synth_minor (42.95) |
| Std copy-paste | 43.36 | [39.31; 48.16] | 0.33 | 62.30 | 46.84 | exp06v11_synth100_real100_s2 (43.26) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | -0.57 | [-2.81; +0.98] | 0.4690 |
| MF clean-bg − Real-Only | AP50 | -0.14 | [-2.77; +1.75] | 0.8650 |
| MF clean-bg − Real-Only | AP75 | -1.31 | [-4.12; +0.55] | 0.1610 |
| Std copy-paste − Real-Only | AP | -0.16 | [-4.53; +1.89] | 0.7010 |
| Std copy-paste − Real-Only | AP50 | -1.27 | [-6.42; +0.91] | 0.2060 |
| Std copy-paste − Real-Only | AP75 | +0.09 | [-4.75; +2.51] | 0.8600 |
| MF clean-bg − Std copy-paste | AP | -0.40 | [-1.60; +2.43] | 0.9140 |
| MF clean-bg − Std copy-paste | AP50 | +1.13 | [-0.36; +4.66] | 0.1210 |
| MF clean-bg − Std copy-paste | AP75 | -1.40 | [-2.98; +1.67] | 0.3430 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | -0.59 | [-2.89; +1.25] | 0.5080 |
| Std copy-paste − Real-Only | 21 | +0.19 | [-4.54; +2.28] | 0.9190 |
| MF clean-bg − Std copy-paste | 21 | -0.78 | [-2.06; +2.71] | 0.6610 |
| MF clean-bg − Real-Only | 22 | -0.37 | [-2.38; +1.21] | 0.6470 |
| Std copy-paste − Real-Only | 22 | -0.10 | [-3.92; +2.17] | 0.8530 |
| MF clean-bg − Std copy-paste | 22 | -0.27 | [-1.72; +2.48] | 0.8930 |
| MF clean-bg − Real-Only | 23 | -0.74 | [-3.50; +0.88] | 0.3900 |
| Std copy-paste − Real-Only | 23 | -0.58 | [-5.32; +1.26] | 0.3990 |
| MF clean-bg − Std copy-paste | 23 | -0.16 | [-1.29; +2.40] | 0.7370 |

Model selected for deployment (best mean AP; seed closest to the mean): Real-Only, exp07_real100.

The CI of the mean reflects the variation of VAL with the 3 seeds used; the SD across seeds reflects the variation of training.
