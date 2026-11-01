# Transfer check: AP on VAL (n = 2185 images, 204 groups of the group partition; group bootstrap, B = 2000)

| Run | AP | AP50 | AP75 | APs | APm | APl | 95% CI AP |
|---|---|---|---|---|---|---|---|
| px_real10 | 33.36 | 52.24 | 35.31 | 0.30 | 21.43 | 39.42 | [29.93; 37.12] |
| px_v9_real10 | 35.00 | 55.05 | 37.45 | 0.74 | 21.40 | 42.07 | [31.24; 39.78] |
| px_p_real10 | 36.61 | 55.06 | 38.46 | 0.80 | 23.51 | 44.21 | [32.45; 41.45] |
| px_q_real10 | 36.09 | 56.33 | 37.60 | 0.85 | 23.43 | 43.46 | [32.11; 40.80] |
| px_v9_real10_s2 | 34.04 | 53.93 | 36.47 | 0.50 | 22.88 | 40.45 | [30.34; 38.17] |
| px_p_real10_s2 | 35.15 | 54.13 | 37.38 | 0.29 | 23.63 | 41.89 | [31.12; 39.22] |

## Paired differences (same group resamples)

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| px_v9_real10 − px_real10 | AP | +1.64 | [+0.06; +4.84] | 0.0420 |
| px_v9_real10 − px_real10 | AP50 | +2.82 | [+0.61; +6.58] | 0.0140 |
| px_v9_real10 − px_real10 | AP75 | +2.15 | [+0.04; +5.66] | 0.0480 |
| px_p_real10 − px_real10 | AP | +3.25 | [+1.26; +6.12] | 0.0000 |
| px_p_real10 − px_real10 | AP50 | +2.83 | [-0.52; +6.95] | 0.0970 |
| px_p_real10 − px_real10 | AP75 | +3.15 | [+0.23; +6.71] | 0.0300 |
| px_q_real10 − px_real10 | AP | +2.74 | [+1.08; +5.37] | 0.0000 |
| px_q_real10 − px_real10 | AP50 | +4.09 | [+1.31; +7.59] | 0.0030 |
| px_q_real10 − px_real10 | AP75 | +2.30 | [+0.19; +5.29] | 0.0340 |
| px_p_real10 − px_v9_real10 | AP | +1.61 | [+0.15; +2.89] | 0.0330 |
| px_p_real10 − px_v9_real10 | AP50 | +0.01 | [-2.72; +2.47] | 0.8790 |
| px_p_real10 − px_v9_real10 | AP75 | +1.01 | [-1.50; +3.00] | 0.5130 |
| px_q_real10 − px_v9_real10 | AP | +1.10 | [-0.67; +2.60] | 0.1930 |
| px_q_real10 − px_v9_real10 | AP50 | +1.27 | [-1.27; +3.55] | 0.3830 |
| px_q_real10 − px_v9_real10 | AP75 | +0.15 | [-2.29; +2.08] | 0.9400 |
| px_q_real10 − px_p_real10 | AP | -0.52 | [-1.83; +0.74] | 0.4660 |
| px_q_real10 − px_p_real10 | AP50 | +1.26 | [-0.74; +2.87] | 0.1790 |
| px_q_real10 − px_p_real10 | AP75 | -0.86 | [-2.78; +1.23] | 0.4820 |
| px_v9_real10_s2 − px_real10 | AP | +0.68 | [-1.11; +2.74] | 0.5430 |
| px_v9_real10_s2 − px_real10 | AP50 | +1.69 | [-1.28; +5.28] | 0.2410 |
| px_v9_real10_s2 − px_real10 | AP75 | +1.16 | [-1.45; +4.40] | 0.4930 |
| px_p_real10_s2 − px_real10 | AP | +1.79 | [-0.14; +3.34] | 0.0710 |
| px_p_real10_s2 − px_real10 | AP50 | +1.89 | [-0.94; +4.67] | 0.2130 |
| px_p_real10_s2 − px_real10 | AP75 | +2.07 | [-0.44; +4.20] | 0.1180 |
| px_p_real10_s2 − px_v9_real10_s2 | AP | +1.11 | [-1.26; +2.95] | 0.2890 |
| px_p_real10_s2 − px_v9_real10_s2 | AP50 | +0.20 | [-3.55; +3.37] | 0.9640 |
| px_p_real10_s2 − px_v9_real10_s2 | AP75 | +0.91 | [-2.87; +3.73] | 0.5170 |

## Seed means per arm

| A − B | Metric | Difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| p_pos_tight (mean s21+s22) − v9 (mean s21+s22) | AP | +1.36 | [-0.27; +2.76] | 0.1070 |
| p_pos_tight (mean s21+s22) − v9 (mean s21+s22) | AP50 | +0.10 | [-2.48; +2.59] | 0.9160 |
| p_pos_tight (mean s21+s22) − v9 (mean s21+s22) | AP75 | +0.96 | [-1.38; +2.67] | 0.4010 |
| p_pos_tight (mean s21+s22) − real (s21) | AP | +2.52 | [+0.68; +4.34] | 0.0040 |
| p_pos_tight (mean s21+s22) − real (s21) | AP50 | +2.36 | [-0.65; +5.35] | 0.1100 |
| p_pos_tight (mean s21+s22) − real (s21) | AP75 | +2.61 | [+0.25; +4.71] | 0.0310 |
| v9 (mean s21+s22) − real (s21) | AP | +1.16 | [-0.43; +3.71] | 0.1820 |
| v9 (mean s21+s22) − real (s21) | AP50 | +2.25 | [-0.16; +5.89] | 0.0660 |
| v9 (mean s21+s22) − real (s21) | AP75 | +1.65 | [-0.57; +5.00] | 0.1580 |

The CI of each run reflects the variation of VAL, not of the seed; the seed mean reduces the latter
but with 2 seeds it does not estimate it (the difference between seeds of the same arm is the noise reference).
