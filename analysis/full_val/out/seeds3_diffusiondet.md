# diffusiondet, 100% real, 3 seeds: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 33.51 | 54.87 | 35.46 | [30.03; 38.23] |
| exp07_real100_s2 | 22 | 33.04 | 56.15 | 34.23 | [29.97; 37.14] |
| exp07_real100_s3 | 23 | 33.34 | 55.67 | 35.24 | [30.06; 38.80] |
| exp08_real_synth_minor | 21 | 35.55 | 59.24 | 37.59 | [31.89; 40.48] |
| exp08_real_synth_minor_s2 | 22 | 34.98 | 57.89 | 37.55 | [31.46; 39.79] |
| exp08_real_synth_minor_s3 | 23 | 34.93 | 57.60 | 37.60 | [31.31; 39.62] |
| exp06v11_synth100_real100 | 21 | 34.80 | 55.07 | 37.80 | [31.01; 39.61] |
| exp06v11_synth100_real100_s2 | 22 | 35.59 | 55.62 | 38.40 | [31.72; 40.90] |
| exp06v11_synth100_real100_s3 | 23 | 35.47 | 55.79 | 38.52 | [31.33; 40.38] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 33.30 | [30.11; 37.81] | 0.24 | 55.56 | 34.98 | exp07_real100_s3 (33.34) |
| MF clean-bg | 35.15 | [31.56; 39.86] | 0.34 | 58.24 | 37.58 | exp08_real_synth_minor_s2 (34.98) |
| Std copy-paste | 35.29 | [31.38; 40.15] | 0.42 | 55.49 | 38.24 | exp06v11_synth100_real100_s3 (35.47) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | +1.85 | [+0.56; +3.07] | 0.0170 |
| MF clean-bg − Real-Only | AP50 | +2.68 | [+0.86; +4.86] | 0.0140 |
| MF clean-bg − Real-Only | AP75 | +2.61 | [+0.52; +4.50] | 0.0160 |
| Std copy-paste − Real-Only | AP | +1.99 | [+0.40; +3.53] | 0.0190 |
| Std copy-paste − Real-Only | AP50 | -0.07 | [-2.41; +1.90] | 0.6940 |
| Std copy-paste − Real-Only | AP75 | +3.26 | [+0.89; +5.14] | 0.0070 |
| MF clean-bg − Std copy-paste | AP | -0.14 | [-1.11; +0.82] | 0.8800 |
| MF clean-bg − Std copy-paste | AP50 | +2.75 | [+1.41; +4.74] | 0.0000 |
| MF clean-bg − Std copy-paste | AP75 | -0.66 | [-2.00; +0.89] | 0.3820 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | +2.04 | [+0.69; +3.75] | 0.0000 |
| Std copy-paste − Real-Only | 21 | +1.29 | [-0.38; +2.83] | 0.1330 |
| MF clean-bg − Std copy-paste | 21 | +0.74 | [-0.41; +2.51] | 0.1980 |
| MF clean-bg − Real-Only | 22 | +1.93 | [+0.45; +4.10] | 0.0090 |
| Std copy-paste − Real-Only | 22 | +2.54 | [+0.77; +5.33] | 0.0050 |
| MF clean-bg − Std copy-paste | 22 | -0.61 | [-2.88; +0.89] | 0.4660 |
| MF clean-bg − Real-Only | 23 | +1.59 | [-1.24; +2.82] | 0.1690 |
| Std copy-paste − Real-Only | 23 | +2.12 | [-0.83; +3.75] | 0.1200 |
| MF clean-bg − Std copy-paste | 23 | -0.53 | [-1.80; +0.74] | 0.4880 |

Model selected for deployment (best mean AP; seed closest to the mean): Std copy-paste, exp06v11_synth100_real100_s3.

The CI of the mean reflects the variation of VAL with the 3 seeds used; the SD across seeds reflects the variation of training.
