# Incremental series: AP on VAL per real fraction (n = 2185 images, 204 groups; group bootstrap, B = 2000)

| Real | Arm | Runs | AP | AP50 | AP75 | 95% CI AP |
|---|---|---|---|---|---|---|
| 10% | Real-Only | px_real10 | 33.36 | 52.24 | 35.31 | [29.93; 37.12] |
| 10% | clean-bg | px_v9_real10 + px_v9_real10_s2 | 34.52 | 54.49 | 36.96 | [30.87; 38.91] |
| 10% | copy-paste | px_p_real10 + px_p_real10_s2 | 35.88 | 54.59 | 37.92 | [31.81; 40.26] |
| 25% | Real-Only | inc_real25 + inc_real25_s2 | 39.04 | 58.59 | 41.39 | [35.03; 44.62] |
| 25% | clean-bg | inc_v9_real25 + inc_v9_real25_s2 | 39.94 | 59.64 | 43.07 | [35.70; 45.13] |
| 25% | copy-paste | inc_v11_real25 + inc_v11_real25_s2 | 39.06 | 57.57 | 41.83 | [34.88; 43.53] |
| 50% | Real-Only | inc_real50 | 40.84 | 60.00 | 43.50 | [36.41; 46.96] |
| 50% | clean-bg | inc_v9_real50 | 41.85 | 61.58 | 44.78 | [37.61; 47.93] |
| 50% | copy-paste | inc_v11_real50 | 40.89 | 59.38 | 43.62 | [36.40; 45.75] |
| 100% | Real-Only | exp07_real100 + exp07_real100_s2 + exp07_real100_s3 | 42.86 | 61.75 | 46.13 | [38.19; 48.16] |
| 100% | clean-bg | exp06_synth100_real100 | 43.55 | 63.64 | 46.53 | [39.22; 48.77] |
| 100% | copy-paste | exp06v11_synth100_real100 + exp06v11_synth100_real100_s2 + exp06v11_synth100_real100_s3 | 42.47 | 60.88 | 45.44 | [37.86; 47.51] |
| 100% | MF clean-bg | exp08_real_synth_minor + exp08_real_synth_minor_s2 + exp08_real_synth_minor_s3 | 43.75 | 63.56 | 46.92 | [39.37; 49.20] |

## Paired differences per fraction (same group resamples)

| Real | A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|---|
| 10% | clean-bg − Real-Only | AP | +1.16 | [-0.43; +3.71] | 0.1820 |
| 10% | clean-bg − Real-Only | AP50 | +2.25 | [-0.16; +5.89] | 0.0660 |
| 10% | clean-bg − Real-Only | AP75 | +1.65 | [-0.57; +5.00] | 0.1580 |
| 10% | copy-paste − Real-Only | AP | +2.52 | [+0.68; +4.34] | 0.0040 |
| 10% | copy-paste − Real-Only | AP50 | +2.36 | [-0.65; +5.35] | 0.1100 |
| 10% | copy-paste − Real-Only | AP75 | +2.61 | [+0.25; +4.71] | 0.0310 |
| 10% | clean-bg − copy-paste | AP | -1.36 | [-2.76; +0.27] | 0.1070 |
| 10% | clean-bg − copy-paste | AP50 | -0.10 | [-2.59; +2.48] | 0.9160 |
| 10% | clean-bg − copy-paste | AP75 | -0.96 | [-2.67; +1.38] | 0.4010 |
| 25% | clean-bg − Real-Only | AP | +0.90 | [-1.44; +3.64] | 0.4290 |
| 25% | clean-bg − Real-Only | AP50 | +1.05 | [-1.82; +4.17] | 0.4690 |
| 25% | clean-bg − Real-Only | AP75 | +1.67 | [-1.29; +4.81] | 0.2710 |
| 25% | copy-paste − Real-Only | AP | +0.03 | [-2.19; +1.32] | 0.7620 |
| 25% | copy-paste − Real-Only | AP50 | -1.03 | [-3.62; +0.54] | 0.1480 |
| 25% | copy-paste − Real-Only | AP75 | +0.44 | [-2.50; +2.06] | 0.9720 |
| 25% | clean-bg − copy-paste | AP | +0.87 | [-0.07; +3.09] | 0.0900 |
| 25% | clean-bg − copy-paste | AP50 | +2.07 | [+0.84; +4.63] | 0.0010 |
| 25% | clean-bg − copy-paste | AP75 | +1.23 | [-0.13; +3.95] | 0.0780 |
| 50% | clean-bg − Real-Only | AP | +1.01 | [-0.85; +2.72] | 0.2670 |
| 50% | clean-bg − Real-Only | AP50 | +1.58 | [-1.10; +3.37] | 0.2710 |
| 50% | clean-bg − Real-Only | AP75 | +1.28 | [-1.39; +3.18] | 0.4070 |
| 50% | copy-paste − Real-Only | AP | +0.05 | [-3.90; +1.20] | 0.7480 |
| 50% | copy-paste − Real-Only | AP50 | -0.62 | [-5.22; +0.76] | 0.3090 |
| 50% | copy-paste − Real-Only | AP75 | +0.12 | [-4.21; +1.86] | 0.7700 |
| 50% | clean-bg − copy-paste | AP | +0.95 | [-0.53; +4.72] | 0.1850 |
| 50% | clean-bg − copy-paste | AP50 | +2.20 | [+0.27; +6.21] | 0.0270 |
| 50% | clean-bg − copy-paste | AP75 | +1.16 | [-0.83; +4.95] | 0.2090 |
| 100% | clean-bg − Real-Only | AP | +0.70 | [-0.69; +2.97] | 0.2800 |
| 100% | clean-bg − Real-Only | AP50 | +1.90 | [-0.38; +4.79] | 0.0920 |
| 100% | clean-bg − Real-Only | AP75 | +0.40 | [-1.77; +3.38] | 0.6830 |
| 100% | copy-paste − Real-Only | AP | -0.39 | [-1.67; +0.51] | 0.3650 |
| 100% | copy-paste − Real-Only | AP50 | -0.86 | [-2.77; +0.36] | 0.1650 |
| 100% | copy-paste − Real-Only | AP75 | -0.70 | [-2.27; +0.54] | 0.2540 |
| 100% | clean-bg − copy-paste | AP | +1.08 | [+0.07; +3.11] | 0.0410 |
| 100% | clean-bg − copy-paste | AP50 | +2.76 | [+1.39; +5.34] | 0.0010 |
| 100% | clean-bg − copy-paste | AP75 | +1.09 | [-0.78; +3.79] | 0.2000 |
| 100% | MF clean-bg − Real-Only | AP | +0.89 | [-0.33; +3.36] | 0.1530 |
| 100% | MF clean-bg − Real-Only | AP50 | +1.82 | [-0.18; +4.60] | 0.0680 |
| 100% | MF clean-bg − Real-Only | AP75 | +0.79 | [-0.70; +3.47] | 0.2670 |
| 100% | MF clean-bg − copy-paste | AP | +1.28 | [+0.21; +3.67] | 0.0190 |
| 100% | MF clean-bg − copy-paste | AP50 | +2.68 | [+1.25; +5.41] | 0.0000 |
| 100% | MF clean-bg − copy-paste | AP75 | +1.49 | [+0.09; +4.21] | 0.0390 |

One seed per arm except where several runs are listed (mean of seeds 21 and 22; at 100%, 21, 22 and 23). The CI reflects the variation of VAL, not of the seed (measured seed-to-seed noise: 1-1.5 AP). At 100% the batches are not matched in pasted objects.
