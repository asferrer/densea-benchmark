# rtdetrv2, 100% real, 3 seeds: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 42.94 | 61.75 | 46.04 | [38.22; 47.95] |
| exp07_real100_s2 | 22 | 43.48 | 62.62 | 46.76 | [38.82; 49.32] |
| exp07_real100_s3 | 23 | 42.15 | 60.87 | 45.60 | [37.56; 47.72] |
| exp08_real_synth_minor | 21 | 44.08 | 63.71 | 47.94 | [39.41; 49.75] |
| exp08_real_synth_minor_s2 | 22 | 43.37 | 63.33 | 46.56 | [39.06; 48.91] |
| exp08_real_synth_minor_s3 | 23 | 43.79 | 63.65 | 46.27 | [39.29; 49.38] |
| exp06v11_synth100_real100 | 21 | 42.66 | 61.10 | 45.52 | [37.96; 47.75] |
| exp06v11_synth100_real100_s2 | 22 | 42.16 | 60.88 | 44.90 | [37.82; 47.48] |
| exp06v11_synth100_real100_s3 | 23 | 42.59 | 60.66 | 45.90 | [37.62; 47.46] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 42.86 | [38.19; 48.16] | 0.67 | 61.75 | 46.13 | exp07_real100 (42.94) |
| MF clean-bg | 43.75 | [39.37; 49.20] | 0.36 | 63.56 | 46.92 | exp08_real_synth_minor_s3 (43.79) |
| Std copy-paste | 42.47 | [37.86; 47.51] | 0.27 | 60.88 | 45.44 | exp06v11_synth100_real100_s3 (42.59) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | +0.89 | [-0.33; +3.36] | 0.1530 |
| MF clean-bg − Real-Only | AP50 | +1.82 | [-0.18; +4.60] | 0.0680 |
| MF clean-bg − Real-Only | AP75 | +0.79 | [-0.70; +3.47] | 0.2670 |
| Std copy-paste − Real-Only | AP | -0.39 | [-1.67; +0.51] | 0.3650 |
| Std copy-paste − Real-Only | AP50 | -0.86 | [-2.77; +0.36] | 0.1650 |
| Std copy-paste − Real-Only | AP75 | -0.70 | [-2.27; +0.54] | 0.2540 |
| MF clean-bg − Std copy-paste | AP | +1.28 | [+0.21; +3.67] | 0.0190 |
| MF clean-bg − Std copy-paste | AP50 | +2.68 | [+1.25; +5.41] | 0.0000 |
| MF clean-bg − Std copy-paste | AP75 | +1.49 | [+0.09; +4.21] | 0.0390 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | +1.14 | [-0.23; +4.05] | 0.0910 |
| Std copy-paste − Real-Only | 21 | -0.29 | [-1.54; +1.17] | 0.6640 |
| MF clean-bg − Std copy-paste | 21 | +1.42 | [+0.20; +3.80] | 0.0240 |
| MF clean-bg − Real-Only | 22 | -0.11 | [-2.25; +3.28] | 0.9160 |
| Std copy-paste − Real-Only | 22 | -1.32 | [-3.24; +0.60] | 0.1690 |
| MF clean-bg − Std copy-paste | 22 | +1.21 | [-0.19; +3.41] | 0.1020 |
| MF clean-bg − Real-Only | 23 | +1.64 | [-0.10; +3.59] | 0.0620 |
| Std copy-paste − Real-Only | 23 | +0.44 | [-2.27; +1.64] | 0.9350 |
| MF clean-bg − Std copy-paste | 23 | +1.20 | [-0.23; +4.71] | 0.0910 |

Model selected for deployment (best mean AP; seed closest to the mean): MF clean-bg, exp08_real_synth_minor_s3.

The CI of the mean reflects the variation of VAL with the 3 seeds used; the SD across seeds reflects the variation of training.
