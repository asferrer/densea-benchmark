# Replication on rfdetr, 100% real: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | Arm | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | Real-Only | 43.54 | 63.68 | 46.93 | [39.45; 49.78] |
| exp08_real_synth_minor | MF clean-bg | 42.95 | 63.47 | 45.32 | [39.01; 48.96] |
| exp06v11_synth100_real100 | Std copy-paste | 43.73 | 62.84 | 47.33 | [39.55; 48.43] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | -0.59 | [-2.89; +1.25] | 0.5080 |
| MF clean-bg − Real-Only | AP50 | -0.21 | [-2.95; +2.02] | 0.7990 |
| MF clean-bg − Real-Only | AP75 | -1.61 | [-4.61; +0.51] | 0.1330 |
| Std copy-paste − Real-Only | AP | +0.19 | [-4.54; +2.28] | 0.9190 |
| Std copy-paste − Real-Only | AP50 | -0.84 | [-6.16; +1.46] | 0.3670 |
| Std copy-paste − Real-Only | AP75 | +0.40 | [-4.90; +2.92] | 0.9510 |
| MF clean-bg − Std copy-paste | AP | -0.78 | [-2.06; +2.71] | 0.6610 |
| MF clean-bg − Std copy-paste | AP50 | +0.63 | [-1.08; +4.74] | 0.3760 |
| MF clean-bg − Std copy-paste | AP75 | -2.01 | [-3.90; +1.81] | 0.2430 |

One seed (21) per arm: the CI reflects the variation of VAL, not of the seed.
