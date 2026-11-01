# DiffusionDet, 100% real, 3 seeds: AP on VAL with 1, 2 and 4 sampling steps (GPU; 204 groups of the group partition; group bootstrap, B = 2000)

| Run | AP 1 step | AP 2 steps | AP 4 steps |
|---|---|---|---|
| exp07_real100 | 33.49 | 33.40 | 33.92 |
| exp07_real100_s2 | 33.14 | 33.13 | 33.24 |
| exp07_real100_s3 | 33.55 | 33.49 | 33.59 |
| exp08_real_synth_minor | 35.42 | 35.45 | 35.62 |
| exp08_real_synth_minor_s2 | 34.94 | 34.88 | 35.08 |
| exp08_real_synth_minor_s3 | 35.09 | 34.95 | 35.16 |
| exp06v11_synth100_real100 | 35.02 | 34.88 | 34.92 |
| exp06v11_synth100_real100_s2 | 35.73 | 35.15 | 35.48 |
| exp06v11_synth100_real100_s3 | 35.36 | 35.57 | 35.56 |

## Mean of 3 seeds per arm

| Arm | AP 1 step | AP50 1 step | AP75 1 step | AP 2 steps | AP50 2 steps | AP75 2 steps | AP 4 steps | AP50 4 steps | AP75 4 steps |
|---|---|---|---|---|---|---|---|---|---|
| Real-Only | 33.39 | 55.74 | 35.12 | 33.34 | 55.68 | 34.84 | 33.58 | 56.23 | 35.13 |
| MF clean-bg | 35.15 | 58.23 | 36.97 | 35.09 | 58.11 | 37.48 | 35.29 | 58.65 | 37.25 |
| Std copy-paste | 35.37 | 55.39 | 38.41 | 35.20 | 55.40 | 38.09 | 35.32 | 55.66 | 38.28 |

## k steps − 1 step (seed mean; same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| Real-Only: 2 steps − 1 step | AP | -0.06 | [-0.33; +0.36] | 0.8550 |
| Real-Only: 2 steps − 1 step | AP50 | -0.06 | [-0.54; +0.47] | 0.8660 |
| Real-Only: 2 steps − 1 step | AP75 | -0.28 | [-0.67; +0.44] | 0.4950 |
| Real-Only: 4 steps − 1 step | AP | +0.19 | [-0.18; +0.48] | 0.2720 |
| Real-Only: 4 steps − 1 step | AP50 | +0.49 | [-0.16; +0.88] | 0.1270 |
| Real-Only: 4 steps − 1 step | AP75 | +0.01 | [-0.55; +0.47] | 0.9950 |
| MF clean-bg: 2 steps − 1 step | AP | -0.06 | [-0.36; +0.13] | 0.5280 |
| MF clean-bg: 2 steps − 1 step | AP50 | -0.11 | [-0.59; +0.20] | 0.5180 |
| MF clean-bg: 2 steps − 1 step | AP75 | +0.51 | [+0.02; +1.04] | 0.0380 |
| MF clean-bg: 4 steps − 1 step | AP | +0.14 | [-0.13; +0.37] | 0.3010 |
| MF clean-bg: 4 steps − 1 step | AP50 | +0.43 | [-0.17; +0.84] | 0.1880 |
| MF clean-bg: 4 steps − 1 step | AP75 | +0.28 | [-0.18; +0.73] | 0.2220 |
| Std copy-paste: 2 steps − 1 step | AP | -0.17 | [-0.61; +0.04] | 0.1110 |
| Std copy-paste: 2 steps − 1 step | AP50 | +0.01 | [-0.60; +0.30] | 0.8240 |
| Std copy-paste: 2 steps − 1 step | AP75 | -0.32 | [-0.93; +0.07] | 0.1030 |
| Std copy-paste: 4 steps − 1 step | AP | -0.05 | [-0.57; +0.32] | 0.9850 |
| Std copy-paste: 4 steps − 1 step | AP50 | +0.26 | [-0.41; +0.66] | 0.3310 |
| Std copy-paste: 4 steps − 1 step | AP75 | -0.13 | [-0.75; +0.33] | 0.6790 |
