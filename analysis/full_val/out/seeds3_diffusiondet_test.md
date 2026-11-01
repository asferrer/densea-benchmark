# diffusiondet, 100% real, 3 seeds: AP on TEST (n = 2207 images, 217 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 33.36 | 55.49 | 35.22 | [31.03; 37.42] |
| exp07_real100_s2 | 22 | 34.81 | 58.69 | 36.80 | [32.28; 39.03] |
| exp07_real100_s3 | 23 | 34.17 | 57.59 | 36.91 | [31.60; 38.47] |
| exp08_real_synth_minor | 21 | 35.39 | 59.41 | 37.27 | [32.32; 40.14] |
| exp08_real_synth_minor_s2 | 22 | 36.08 | 60.11 | 39.10 | [33.11; 40.71] |
| exp08_real_synth_minor_s3 | 23 | 36.67 | 61.98 | 38.88 | [33.33; 41.00] |
| exp06v11_synth100_real100 | 21 | 35.31 | 56.14 | 38.01 | [32.75; 39.11] |
| exp06v11_synth100_real100_s2 | 22 | 36.06 | 57.42 | 39.38 | [33.60; 40.21] |
| exp06v11_synth100_real100_s3 | 23 | 34.85 | 54.90 | 38.13 | [32.13; 38.69] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 34.11 | [31.83; 38.08] | 0.73 | 57.26 | 36.31 | exp07_real100_s3 (34.17) |
| MF clean-bg | 36.05 | [33.09; 40.49] | 0.64 | 60.50 | 38.41 | exp08_real_synth_minor_s2 (36.08) |
| Std copy-paste | 35.41 | [32.92; 39.21] | 0.61 | 56.15 | 38.50 | exp06v11_synth100_real100 (35.31) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | +1.93 | [+0.09; +3.58] | 0.0360 |
| MF clean-bg − Real-Only | AP50 | +3.25 | [+0.33; +6.01] | 0.0350 |
| MF clean-bg − Real-Only | AP75 | +2.10 | [-0.41; +4.95] | 0.1160 |
| Std copy-paste − Real-Only | AP | +1.29 | [-0.37; +2.31] | 0.1290 |
| Std copy-paste − Real-Only | AP50 | -1.10 | [-3.63; +0.64] | 0.1740 |
| Std copy-paste − Real-Only | AP75 | +2.19 | [-0.02; +3.85] | 0.0540 |
| MF clean-bg − Std copy-paste | AP | +0.64 | [-1.19; +2.85] | 0.4720 |
| MF clean-bg − Std copy-paste | AP50 | +4.35 | [+1.25; +7.72] | 0.0030 |
| MF clean-bg − Std copy-paste | AP75 | -0.09 | [-2.55; +3.23] | 0.9930 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | +2.03 | [-0.28; +4.44] | 0.1010 |
| Std copy-paste − Real-Only | 21 | +1.95 | [-0.25; +3.41] | 0.0800 |
| MF clean-bg − Std copy-paste | 21 | +0.08 | [-1.64; +2.79] | 0.8490 |
| MF clean-bg − Real-Only | 22 | +1.27 | [-0.78; +3.24] | 0.2210 |
| Std copy-paste − Real-Only | 22 | +1.25 | [-0.66; +2.79] | 0.1840 |
| MF clean-bg − Std copy-paste | 22 | +0.02 | [-2.30; +2.57] | 0.9720 |
| MF clean-bg − Real-Only | 23 | +2.51 | [-0.28; +4.71] | 0.0830 |
| Std copy-paste − Real-Only | 23 | +0.68 | [-1.68; +2.21] | 0.6830 |
| MF clean-bg − Std copy-paste | 23 | +1.82 | [-0.39; +4.08] | 0.1110 |

The CI of the mean reflects the variation of TEST with the 3 seeds used; the SD across seeds reflects the variation of training.
