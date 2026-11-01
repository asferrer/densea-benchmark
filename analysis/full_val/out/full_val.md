# 100% real: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|
| exp07_real100 | 42.94 | 61.75 | 46.04 | [38.22; 47.95] |
| exp06_synth100_real100 | 43.55 | 63.64 | 46.53 | [39.22; 48.77] |
| exp08_real_synth_minor | 44.08 | 63.71 | 47.94 | [39.41; 49.75] |
| exp09_real_synth_invprop | 43.97 | 63.38 | 47.58 | [39.68; 49.64] |
| exp06v11_synth100_real100 | 42.66 | 61.10 | 45.52 | [37.96; 47.75] |
| exp08v11_real_synth_minor | 42.01 | 60.79 | 44.55 | [37.53; 46.85] |
| exp09v11_real_synth_invprop | 42.57 | 61.57 | 45.52 | [38.00; 47.45] |
| exp07_real100_s2 | 43.48 | 62.62 | 46.76 | [38.82; 49.32] |
| exp08_real_synth_minor_s2 | 43.37 | 63.33 | 46.56 | [39.06; 48.91] |
| exp06v11_synth100_real100_s2 | 42.16 | 60.88 | 44.90 | [37.82; 47.48] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| exp06_synth100_real100 − exp07_real100 | AP | +0.61 | [-0.62; +3.15] | 0.3010 |
| exp06_synth100_real100 − exp07_real100 | AP50 | +1.90 | [+0.08; +5.06] | 0.0460 |
| exp06_synth100_real100 − exp07_real100 | AP75 | +0.49 | [-1.77; +3.90] | 0.5440 |
| exp08_real_synth_minor − exp07_real100 | AP | +1.14 | [-0.23; +4.05] | 0.0910 |
| exp08_real_synth_minor − exp07_real100 | AP50 | +1.97 | [+0.36; +5.31] | 0.0280 |
| exp08_real_synth_minor − exp07_real100 | AP75 | +1.90 | [-0.05; +5.31] | 0.0570 |
| exp09_real_synth_invprop − exp07_real100 | AP | +1.03 | [-0.29; +4.04] | 0.1100 |
| exp09_real_synth_invprop − exp07_real100 | AP50 | +1.64 | [-0.19; +5.19] | 0.0770 |
| exp09_real_synth_invprop − exp07_real100 | AP75 | +1.54 | [-0.42; +5.21] | 0.0960 |
| exp06v11_synth100_real100 − exp07_real100 | AP | -0.29 | [-1.54; +1.17] | 0.6640 |
| exp06v11_synth100_real100 − exp07_real100 | AP50 | -0.64 | [-2.34; +1.23] | 0.5110 |
| exp06v11_synth100_real100 − exp07_real100 | AP75 | -0.52 | [-2.18; +1.30] | 0.5470 |
| exp08v11_real_synth_minor − exp07_real100 | AP | -0.93 | [-2.31; +0.23] | 0.1250 |
| exp08v11_real_synth_minor − exp07_real100 | AP50 | -0.95 | [-2.66; +0.58] | 0.2090 |
| exp08v11_real_synth_minor − exp07_real100 | AP75 | -1.49 | [-3.45; +0.22] | 0.0860 |
| exp09v11_real_synth_invprop − exp07_real100 | AP | -0.38 | [-1.89; +0.77] | 0.5740 |
| exp09v11_real_synth_invprop − exp07_real100 | AP50 | -0.18 | [-1.88; +1.25] | 0.7570 |
| exp09v11_real_synth_invprop − exp07_real100 | AP75 | -0.51 | [-2.65; +1.05] | 0.5480 |
| exp06_synth100_real100 − exp06v11_synth100_real100 | AP | +0.90 | [-0.10; +2.86] | 0.0730 |
| exp06_synth100_real100 − exp06v11_synth100_real100 | AP50 | +2.54 | [+1.29; +4.88] | 0.0000 |
| exp06_synth100_real100 − exp06v11_synth100_real100 | AP75 | +1.01 | [-1.02; +3.69] | 0.2580 |
| exp08_real_synth_minor − exp08v11_real_synth_minor | AP | +2.07 | [+0.53; +5.32] | 0.0150 |
| exp08_real_synth_minor − exp08v11_real_synth_minor | AP50 | +2.92 | [+1.09; +6.66] | 0.0020 |
| exp08_real_synth_minor − exp08v11_real_synth_minor | AP75 | +3.39 | [+1.32; +7.27] | 0.0000 |
| exp09_real_synth_invprop − exp09v11_real_synth_invprop | AP | +1.40 | [-0.16; +4.73] | 0.0650 |
| exp09_real_synth_invprop − exp09v11_real_synth_invprop | AP50 | +1.81 | [+0.05; +5.41] | 0.0460 |
| exp09_real_synth_invprop − exp09v11_real_synth_invprop | AP75 | +2.06 | [-0.10; +6.33] | 0.0570 |
| exp08_real_synth_minor − exp06v11_synth100_real100 | AP | +1.42 | [+0.20; +3.80] | 0.0240 |
| exp08_real_synth_minor − exp06v11_synth100_real100 | AP50 | +2.61 | [+1.07; +5.34] | 0.0020 |
| exp08_real_synth_minor − exp06v11_synth100_real100 | AP75 | +2.42 | [+0.59; +5.23] | 0.0090 |
| exp08_real_synth_minor_s2 − exp07_real100_s2 | AP | -0.11 | [-2.25; +3.28] | 0.9160 |
| exp08_real_synth_minor_s2 − exp07_real100_s2 | AP50 | +0.71 | [-2.84; +4.61] | 0.6400 |
| exp08_real_synth_minor_s2 − exp07_real100_s2 | AP75 | -0.20 | [-2.62; +3.40] | 0.8770 |
| exp06v11_synth100_real100_s2 − exp07_real100_s2 | AP | -1.32 | [-3.24; +0.60] | 0.1690 |
| exp06v11_synth100_real100_s2 − exp07_real100_s2 | AP50 | -1.74 | [-5.48; +0.80] | 0.1750 |
| exp06v11_synth100_real100_s2 − exp07_real100_s2 | AP75 | -1.87 | [-4.13; +0.63] | 0.1370 |
| exp08_real_synth_minor_s2 − exp06v11_synth100_real100_s2 | AP | +1.21 | [-0.19; +3.41] | 0.1020 |
| exp08_real_synth_minor_s2 − exp06v11_synth100_real100_s2 | AP50 | +2.45 | [+0.72; +5.11] | 0.0040 |
| exp08_real_synth_minor_s2 − exp06v11_synth100_real100_s2 | AP75 | +1.67 | [-0.41; +4.30] | 0.1170 |

## Seed means per arm

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| MF clean-bg (mean) − Real-Only (mean) | AP | +0.51 | [-0.82; +3.51] | 0.4510 |
| MF clean-bg (mean) − Real-Only (mean) | AP50 | +1.34 | [-0.77; +4.80] | 0.1960 |
| MF clean-bg (mean) − Real-Only (mean) | AP75 | +0.85 | [-0.77; +4.19] | 0.2850 |
| Std copy-paste (mean) − Real-Only (mean) | AP | -0.80 | [-2.08; +0.74] | 0.2440 |
| Std copy-paste (mean) − Real-Only (mean) | AP50 | -1.19 | [-3.50; +0.74] | 0.2050 |
| Std copy-paste (mean) − Real-Only (mean) | AP75 | -1.20 | [-2.80; +0.69] | 0.1690 |
| MF clean-bg (mean) − Std copy-paste (mean) | AP | +1.32 | [+0.32; +3.52] | 0.0120 |
| MF clean-bg (mean) − Std copy-paste (mean) | AP50 | +2.53 | [+1.22; +5.05] | 0.0000 |
| MF clean-bg (mean) − Std copy-paste (mean) | AP75 | +2.04 | [+0.53; +4.48] | 0.0100 |

The CI of each run reflects the variation of VAL, not of the seed. Runs without the _s2 suffix are seed 21.
