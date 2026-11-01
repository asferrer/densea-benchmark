# Replication on diffusiondet, 100% real: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | Arm | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | Real-Only | 33.51 | 54.87 | 35.46 | [30.03; 38.23] |
| exp08_real_synth_minor | MF clean-bg | 35.55 | 59.24 | 37.59 | [31.89; 40.48] |
| exp06v11_synth100_real100 | Std copy-paste | 34.80 | 55.07 | 37.80 | [31.01; 39.61] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | +2.04 | [+0.69; +3.75] | 0.0000 |
| MF clean-bg − Real-Only | AP50 | +4.37 | [+1.93; +7.62] | 0.0000 |
| MF clean-bg − Real-Only | AP75 | +2.13 | [+0.01; +4.62] | 0.0490 |
| Std copy-paste − Real-Only | AP | +1.29 | [-0.38; +2.83] | 0.1330 |
| Std copy-paste − Real-Only | AP50 | +0.20 | [-2.66; +2.30] | 0.9080 |
| Std copy-paste − Real-Only | AP75 | +2.34 | [-0.48; +4.26] | 0.1050 |
| MF clean-bg − Std copy-paste | AP | +0.74 | [-0.41; +2.51] | 0.1980 |
| MF clean-bg − Std copy-paste | AP50 | +4.17 | [+2.23; +7.43] | 0.0000 |
| MF clean-bg − Std copy-paste | AP75 | -0.21 | [-1.86; +2.57] | 1.0000 |

One seed (21) per arm: the CI reflects the variation of VAL, not of the seed.
