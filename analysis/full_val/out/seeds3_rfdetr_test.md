# rfdetr, 100% real, 3 seeds: AP on TEST (n = 2207 images, 217 groups of the group partition; group bootstrap, B = 2000)

| Run | Seed | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|
| exp07_real100 | 21 | 44.55 | 65.61 | 47.77 | [40.84; 50.52] |
| exp07_real100_s2 | 22 | 44.60 | 65.70 | 47.60 | [40.76; 50.39] |
| exp07_real100_s3 | 23 | 44.27 | 64.60 | 47.88 | [40.48; 50.11] |
| exp08_real_synth_minor | 21 | 43.03 | 64.18 | 45.98 | [39.75; 48.11] |
| exp08_real_synth_minor_s2 | 22 | 42.93 | 64.11 | 46.17 | [39.64; 47.90] |
| exp08_real_synth_minor_s3 | 23 | 43.37 | 64.30 | 46.07 | [40.15; 48.29] |
| exp06v11_synth100_real100 | 21 | 43.78 | 64.10 | 46.89 | [40.85; 48.50] |
| exp06v11_synth100_real100_s2 | 22 | 43.08 | 63.24 | 46.24 | [40.12; 47.59] |
| exp06v11_synth100_real100_s3 | 23 | 43.91 | 64.31 | 46.76 | [40.94; 48.45] |

## Mean of 3 seeds per arm

| Arm | Mean AP | 95% CI | SD across seeds | Mean AP50 | Mean AP75 | Seed closest to the mean |
|---|---|---|---|---|---|---|
| Real-Only | 44.47 | [40.74; 50.35] | 0.18 | 65.30 | 47.75 | exp07_real100 (44.55) |
| MF clean-bg | 43.11 | [39.88; 48.01] | 0.23 | 64.20 | 46.07 | exp08_real_synth_minor (43.03) |
| Std copy-paste | 43.59 | [40.67; 48.13] | 0.44 | 63.89 | 46.63 | exp06v11_synth100_real100 (43.78) |

## Differences between seed means (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | AP | -1.36 | [-3.21; +0.02] | 0.0540 |
| MF clean-bg − Real-Only | AP50 | -1.11 | [-3.44; +0.75] | 0.2350 |
| MF clean-bg − Real-Only | AP75 | -1.68 | [-3.97; +0.03] | 0.0540 |
| Std copy-paste − Real-Only | AP | -0.88 | [-3.18; +0.88] | 0.3530 |
| Std copy-paste − Real-Only | AP50 | -1.42 | [-4.10; +0.66] | 0.1730 |
| Std copy-paste − Real-Only | AP75 | -1.12 | [-3.91; +0.99] | 0.3170 |
| MF clean-bg − Std copy-paste | AP | -0.48 | [-1.56; +0.78] | 0.4510 |
| MF clean-bg − Std copy-paste | AP50 | +0.31 | [-1.30; +1.96] | 0.7220 |
| MF clean-bg − Std copy-paste | AP75 | -0.56 | [-1.99; +1.21] | 0.5680 |

## Differences within each seed (AP)

| A − B | Seed | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg − Real-Only | 21 | -1.52 | [-3.20; -0.20] | 0.0200 |
| Std copy-paste − Real-Only | 21 | -0.77 | [-2.97; +0.99] | 0.4530 |
| MF clean-bg − Std copy-paste | 21 | -0.75 | [-1.99; +0.62] | 0.2560 |
| MF clean-bg − Real-Only | 22 | -1.67 | [-3.76; +0.11] | 0.0710 |
| Std copy-paste − Real-Only | 22 | -1.52 | [-3.88; +0.41] | 0.1390 |
| MF clean-bg − Std copy-paste | 22 | -0.15 | [-1.28; +1.29] | 0.9090 |
| MF clean-bg − Real-Only | 23 | -0.90 | [-2.88; +0.45] | 0.2330 |
| Std copy-paste − Real-Only | 23 | -0.36 | [-2.87; +1.43] | 0.6580 |
| MF clean-bg − Std copy-paste | 23 | -0.54 | [-1.65; +0.82] | 0.4530 |

The CI of the mean reflects the variation of TEST with the 3 seeds used; the SD across seeds reflects the variation of training.
