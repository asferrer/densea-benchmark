# Panel appendix: task evaluator (RT-DETRv2 with 10 % real)

## Summary of the findings for the task evaluator

**What was measured.** RT-DETRv2 R50 of `px_real10/best.pth` (10 % real TRAIN, EMA weights, `last_epoch` 187 of the checkpoint) on CPU and on the host (no container, no GPU). Each object is the crop with context (box +20 % per side, like the other extractors) squashed to 640x640 as in VAL (without normalization, as in training), and its vector is the **mean of the P3 map of the HybridEncoder (stride 8, 256 d) inside the object box**. P3 is the highest-resolution map consumed by the decoder, it already mixes backbone and context through the top-down and bottom-up branches, and it is the last thing the detector sees before deciding class and box. Sensitivity in the same pass: P5 of the encoder (stride 32, 256 d) and C5 of the backbone (2048 d). Same sets, quotas, transfer-check batches and rule as the panel; own cache in `cache/gap_panel_task/`.

**1. Does it order the variants like the other extractors?** Only with the grouped AUC, and partially.
- AUC: Spearman of ranks with ResNet-50 0.63, DINOv2 0.05, SigLIP 2 0.82, DINOv3 0.06 and CLIP 0.86; with the mean rank of the 5, 0.59 (among the 5 general ones the mean Spearman is 0.53). It falls within their dispersion.
- KID, CMMD and FD: the 5 general ones agree among themselves (mean Spearman 0.82 / 0.86 / 0.84) and the task evaluator separates from them (with the mean rank of the 5: 0.31 / 0.50 / −0.15). For example, the task evaluator ranks `g_v11_copypaste` worst in KID, CMMD and FD and `p_pos_tight` best, whereas the general ones rank `m_pos`, `j_v11_visible` and `g_v11_copypaste` first. What it shares with all: c_v10_noaug among the worst (rank 9-11 in every metric), ref_batch_v9 with AUC 0.989 and Synthetic Seabed with AUC 1.000.
- It has its own signal and is not a replica of the general ones. With 3 batches with AP it cannot be said which of the two readings is right (see the validation point).

**2. Does it separate v9 from v11?** Yes, but not better than the others.
- Pilot: AUC v9 0.989 against 0.826 of p_pos_tight (−0.163; combined noise 0.049; disjoint CIs). Transfer check: 0.992 against 0.870 of px_p (−0.123; noise 0.029) and 0.867 of px_q. All 6 extractors separate with disjoint CIs.
- In noise units (dAUC/noise, px_p − px_v9): task 4.2; ResNet-50 6.0; DINOv2 5.4; SigLIP 2 5.5; DINOv3 4.9; CLIP 4.1. It leaves more AUC gap than the ViTs (px_p 0.870 against 0.928-0.941 with DINOv2, SigLIP 2, DINOv3 and CLIP; ResNet-50 gives 0.850), but its CIs are wider and it does not win in noise units.

**3. Does it discriminate better among the v11 variants or between p and q?** No; it is the worst.
- Among the 11 variants of the main scenario: 11/55 pairs with disjoint CIs (general ones 10-15/55) and dispersion of the AUC across variants of 0.93 CI half-widths (general ones 1.06-1.47): the smallest of the 6.
- p against q: dAUC (q − p) −0.003 in the transfer-check batches (noise 0.041) and −0.006 in the pilot (noise 0.070); overlapping CIs, as in the 5 general ones (which give −0.012 to −0.022 and −0.004 to −0.040, also without separating).
- The alternative layers give the same order (Spearman of AUC ranks with P3: 0.88 with P5 and 0.90 with C5).

**4. Does the verdict of s_drop change with the adoption rule?** No. `s_drop` is still ADOPTED: with the task evaluator AUC 0.822 [0.777; 0.867] against 0.833 [0.795; 0.875] of the control (dAUC −0.011, inside the CI), Vendi 2.5 / 2.5 and coverage@val 0.69 / 0.69. With 6 extractors: 6/6 not worse and 6/6 with diversity not lower (minimum 3). P5 and C5 agree.

**Validation against the AP.** With 3 batches (5 arms) the task evaluator agrees in 7/8 pairs (5/5 with |dAP| ≥ 1) and Spearman 0.79, identical to the general ones. Like the others, it orders px_v9 behind px_p and px_q; with the AUC it places q ahead of p (the AP leaves them p 36.61 and q 36.09, within the noise of 1-1.5 AP) and with the CMMD p ahead of q.

**Caveats.**
- 111 of the 976 real TRAIN crops of the reference (11.4 %) come from images that were part of the 10 % real subset with which the detector was trained (0 of the 861 of VAL); the influence has not been measured. The TRAIN-VAL floor gives AUC 0.455, as with the other extractors.
- Dimension 256 against 512-2048 of the general ones and a single extraction seed: the scale of KID and FD is not comparable across extractors.
- A single detector seed and a single detector (10 % real): the evaluator measures what that model encodes.


## Extractors of the appendix

| Extractor | Weights | Dim | Notes |
|---|---|---|---|
| rtdetrv2_real10 | px_real10/best.pth | 256 | HybridEncoder P3 (stride 8, 256 d), mean inside the object box; 640x640 input without normalization |
| rtdetrv2_real10_p5 | px_real10/best.pth | 256 | sensitivity: HybridEncoder P5 (stride 32, 256 d), mean inside the box |
| rtdetrv2_real10_c5 | px_real10/best.pth | 2048 | sensitivity: ResNet-50-vd backbone C5 (stride 32, 2048 d), mean inside the box |

## Panel with the main extractor (TRAIN and VAL reference)

### TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.455 ± 0.017 [0.377; 0.535] | 1.5 | 0.0022 [0.0015; 0.0095] | 0.018 | 0.95 [0.95; 0.97] | 0.93 [0.90; 0.95] | 0.93 [0.88; 1.00] | 0.89 [0.80; 0.92] | 2.4 | 0.049 | 0.667 [0.59; 0.75] |
| ref_batch_v9 | 454 | 0.989 ± 0.001 [0.984; 0.995] | 8.8 | 0.0833 [0.0775; 0.0892] | 0.536 | 0.93 [0.87; 0.96] | 0.23 [0.19; 0.30] | 0.75 [0.67; 0.82] | 0.47 [0.44; 0.54] | 2.2 | 0.052 | 0.461 [0.38; 0.55] |
| c_v10_noaug | 454 | 0.954 ± 0.002 [0.939; 0.972] | 5.1 | 0.0396 [0.0352; 0.0436] | 0.261 | 0.97 [0.94; 0.99] | 0.49 [0.45; 0.55] | 1.05 [0.97; 1.17] | 0.72 [0.69; 0.79] | 2.2 | 0.051 | 0.569 [0.49; 0.64] |
| g_v11_copypaste | 454 | 0.896 ± 0.002 [0.872; 0.923] | 5.9 | 0.0452 [0.0416; 0.0526] | 0.261 | 0.94 [0.85; 0.95] | 0.72 [0.68; 0.77] | 0.75 [0.66; 0.86] | 0.68 [0.63; 0.74] | 2.7 | 0.065 | 0.540 [0.47; 0.61] |
| j_v11_visible | 454 | 0.866 ± 0.008 [0.837; 0.899] | 4.6 | 0.0295 [0.0251; 0.0368] | 0.172 | 0.93 [0.89; 0.96] | 0.72 [0.67; 0.76] | 0.84 [0.78; 0.94] | 0.67 [0.66; 0.73] | 2.6 | 0.063 | 0.506 [0.41; 0.57] |
| m_pos | 454 | 0.871 ± 0.005 [0.846; 0.903] | 5.2 | 0.0369 [0.0312; 0.0431] | 0.207 | 0.89 [0.85; 0.92] | 0.70 [0.69; 0.76] | 0.77 [0.72; 0.85] | 0.69 [0.65; 0.75] | 2.6 | 0.064 | 0.510 [0.44; 0.59] |
| p_pos_tight | 454 | 0.826 ± 0.007 [0.776; 0.874] | 3.4 | 0.0195 [0.0149; 0.0260] | 0.118 | 0.96 [0.93; 0.98] | 0.73 [0.72; 0.78] | 0.94 [0.88; 1.03] | 0.78 [0.76; 0.82] | 2.5 | 0.058 | 0.534 [0.47; 0.62] |
| q_base | 454 | 0.844 ± 0.002 [0.802; 0.887] | 4.6 | 0.0331 [0.0284; 0.0422] | 0.200 | 0.94 [0.91; 0.96] | 0.76 [0.70; 0.79] | 0.92 [0.85; 1.00] | 0.74 [0.71; 0.78] | 2.6 | 0.061 | 0.563 [0.49; 0.63] |
| q_lab | 454 | 0.833 ± 0.007 [0.786; 0.883] | 4.8 | 0.0358 [0.0284; 0.0509] | 0.214 | 0.96 [0.94; 0.97] | 0.72 [0.69; 0.78] | 0.92 [0.86; 1.04] | 0.76 [0.74; 0.80] | 2.5 | 0.057 | 0.523 [0.45; 0.59] |
| q_lab_same | 454 | 0.820 ± 0.004 [0.770; 0.871] | 4.8 | 0.0362 [0.0280; 0.0486] | 0.219 | 0.94 [0.93; 0.97] | 0.79 [0.77; 0.84] | 0.90 [0.86; 1.01] | 0.77 [0.74; 0.79] | 2.6 | 0.059 | 0.506 [0.44; 0.58] |
| q_lab_local | 454 | 0.833 ± 0.005 [0.785; 0.885] | 4.4 | 0.0320 [0.0289; 0.0419] | 0.194 | 0.93 [0.90; 0.96] | 0.74 [0.67; 0.78] | 0.91 [0.84; 1.03] | 0.76 [0.72; 0.79] | 2.6 | 0.061 | 0.557 [0.49; 0.65] |
| r_matte | 454 | 0.849 ± 0.005 [0.794; 0.902] | 5.0 | 0.0414 [0.0344; 0.0504] | 0.243 | 0.91 [0.89; 0.95] | 0.72 [0.68; 0.77] | 0.83 [0.77; 0.94] | 0.69 [0.66; 0.75] | 2.6 | 0.063 | 0.493 [0.42; 0.56] |
| r_medium_harm | 454 | 0.826 ± 0.001 [0.772; 0.873] | 4.7 | 0.0367 [0.0288; 0.0508] | 0.218 | 0.93 [0.92; 0.96] | 0.78 [0.75; 0.82] | 0.87 [0.81; 0.98] | 0.75 [0.72; 0.81] | 2.6 | 0.060 | 0.523 [0.45; 0.58] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [0.999; 1.000] | 12.6 | 0.1217 [0.1162; 0.1345] | 0.786 | 0.32 [0.32; 0.54] | 0.07 [0.03; 0.10] | 0.09 [0.09; 0.18] | 0.15 [0.15; 0.21] | 1.8 | 0.039 | 0.335 [0.24; 0.38] |

### VAL reference

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 1.5 | 0.0022 | 0.018 | 0.93 | 0.95 | 0.79 | 0.88 | 0.051 |
| ref_batch_v9 | 9.0 | 0.0825 | 0.555 | 0.89 | 0.19 | 0.71 | 0.34 | 0.070 |
| c_v10_noaug | 5.7 | 0.0473 | 0.309 | 0.93 | 0.57 | 0.94 | 0.56 | 0.058 |
| g_v11_copypaste | 5.8 | 0.0404 | 0.253 | 0.82 | 0.75 | 0.66 | 0.62 | 0.066 |
| j_v11_visible | 4.7 | 0.0272 | 0.171 | 0.85 | 0.72 | 0.71 | 0.58 | 0.061 |
| m_pos | 5.4 | 0.0370 | 0.227 | 0.82 | 0.72 | 0.68 | 0.56 | 0.065 |
| p_pos_tight | 3.1 | 0.0151 | 0.104 | 0.92 | 0.76 | 0.81 | 0.70 | 0.055 |
| q_base | 4.1 | 0.0258 | 0.171 | 0.91 | 0.78 | 0.81 | 0.68 | 0.058 |
| q_lab | 4.4 | 0.0293 | 0.190 | 0.93 | 0.77 | 0.81 | 0.69 | 0.055 |
| q_lab_same | 4.4 | 0.0296 | 0.194 | 0.93 | 0.79 | 0.79 | 0.67 | 0.056 |
| q_lab_local | 3.9 | 0.0260 | 0.171 | 0.92 | 0.76 | 0.81 | 0.68 | 0.055 |
| r_matte | 4.7 | 0.0353 | 0.221 | 0.91 | 0.77 | 0.79 | 0.63 | 0.057 |
| r_medium_harm | 4.2 | 0.0300 | 0.192 | 0.92 | 0.75 | 0.76 | 0.66 | 0.057 |
| Synthetic Seabed (7 classes) | 12.8 | 0.1197 | 0.788 | 0.44 | 0.06 | 0.13 | 0.13 | 0.081 |

### Per-class AUC

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.44 | 0.54 | 0.58 | 0.29 | 0.32 | 0.57 | 0.40 | 0.51 |
| ref_batch_v9 | 0.99 | 0.98 | 0.99 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| c_v10_noaug | 0.96 | 0.94 | 0.97 | 0.99 | 0.97 | 0.89 | 0.98 | 0.96 |
| g_v11_copypaste | 0.89 | 0.92 | 0.96 | 0.94 | 0.93 | 0.82 | 0.87 | 0.88 |
| j_v11_visible | 0.83 | 0.84 | 0.97 | 0.95 | 0.89 | 0.81 | 0.84 | 0.86 |
| m_pos | 0.81 | 0.81 | 0.98 | 0.94 | 0.89 | 0.84 | 0.92 | 0.88 |
| p_pos_tight | 0.77 | 0.73 | 0.97 | 0.94 | 0.84 | 0.78 | 0.87 | 0.85 |
| q_base | 0.85 | 0.73 | 0.96 | 0.92 | 0.85 | 0.82 | 0.86 | 0.87 |
| q_lab | 0.83 | 0.85 | 0.96 | 0.93 | 0.85 | 0.77 | 0.84 | 0.84 |
| q_lab_same | 0.84 | 0.81 | 0.92 | 0.91 | 0.82 | 0.76 | 0.82 | 0.86 |
| q_lab_local | 0.85 | 0.77 | 0.98 | 0.91 | 0.85 | 0.77 | 0.82 | 0.86 |
| r_matte | 0.83 | 0.79 | 0.88 | 0.94 | 0.93 | 0.81 | 0.84 | 0.87 |
| r_medium_harm | 0.82 | 0.82 | 0.95 | 0.92 | 0.85 | 0.74 | 0.83 | 0.84 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### Transfer-check batches

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 861 | 0.434 ± 0.011 [0.367; 0.497] | 1.1 | 0.0033 [0.0031; 0.0096] | 0.021 | 0.93 [0.91; 0.95] | 0.87 [0.85; 0.89] | 0.95 [0.87; 1.01] | 0.84 [0.73; 0.88] | 2.5 | 0.038 | 0.710 [0.66; 0.78] |
| px_v9 | 976 | 0.992 ± 0.000 [0.988; 0.996] | 8.0 | 0.0790 [0.0762; 0.0831] | 0.511 | 0.90 [0.87; 0.93] | 0.16 [0.15; 0.21] | 0.74 [0.66; 0.81] | 0.38 [0.37; 0.43] | 2.2 | 0.046 | 0.540 [0.47; 0.60] |
| px_p | 976 | 0.870 ± 0.003 [0.841; 0.900] | 2.6 | 0.0159 [0.0147; 0.0210] | 0.084 | 0.89 [0.88; 0.93] | 0.73 [0.71; 0.76] | 0.79 [0.76; 0.88] | 0.74 [0.71; 0.78] | 2.6 | 0.053 | 0.603 [0.53; 0.66] |
| px_q | 976 | 0.867 ± 0.002 [0.837; 0.895] | 3.6 | 0.0280 [0.0249; 0.0379] | 0.159 | 0.93 [0.90; 0.94] | 0.72 [0.70; 0.76] | 0.82 [0.78; 0.87] | 0.72 [0.71; 0.76] | 2.6 | 0.052 | 0.610 [0.55; 0.67] |

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 1.1 | 0.0033 | 0.021 | 0.87 | 0.93 | 0.69 | 0.81 | 0.046 |
| px_v9 | 8.1 | 0.0730 | 0.500 | 0.81 | 0.16 | 0.53 | 0.28 | 0.063 |
| px_p | 2.8 | 0.0143 | 0.089 | 0.86 | 0.77 | 0.65 | 0.60 | 0.053 |
| px_q | 3.5 | 0.0243 | 0.151 | 0.89 | 0.77 | 0.66 | 0.59 | 0.052 |

## Does it order the variants like the other extractors?

### Grouped AUC: rank of each variant (1 = best synthetic quality)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 | rtdetrv2_real10 | mean of the 5 |
|---|---|---|---|---|---|---|---|
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 | 11 | 11.0 |
| g_v11_copypaste | 10 | 4 | 10 | 2 | 10 | 10 | 7.2 |
| j_v11_visible | 5 | 1 | 5 | 1 | 5 | 8 | 3.4 |
| m_pos | 3 | 2 | 9 | 4 | 9 | 9 | 5.4 |
| p_pos_tight | 8 | 9 | 8 | 6 | 7 | 3 | 7.6 |
| q_base | 7 | 8 | 7 | 10 | 6 | 6 | 7.6 |
| q_lab | 4 | 6 | 3 | 8 | 4 | 5 | 5.0 |
| q_lab_same | 2 | 5 | 1 | 3 | 2 | 1 | 2.6 |
| q_lab_local | 6 | 7 | 4 | 5 | 3 | 4 | 5.0 |
| r_matte | 9 | 10 | 6 | 9 | 8 | 7 | 8.4 |
| r_medium_harm | 1 | 3 | 2 | 7 | 1 | 2 | 2.8 |

Spearman of `rtdetrv2_real10` with each extractor: resnet50 0.63, dinov2_b14 0.05, siglip2_b16 0.82, dinov3_b16 0.06, clip_b16 0.86. With the mean rank of the 5: 0.59. Mean Spearman among the 5 general ones: 0.53.

### KID: rank of each variant (1 = best synthetic quality)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 | rtdetrv2_real10 | mean of the 5 |
|---|---|---|---|---|---|---|---|
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 | 9 | 11.0 |
| g_v11_copypaste | 3 | 4 | 2 | 3 | 2 | 11 | 2.8 |
| j_v11_visible | 2 | 1 | 3 | 2 | 3 | 2 | 2.2 |
| m_pos | 1 | 2 | 1 | 1 | 1 | 8 | 1.2 |
| p_pos_tight | 5 | 9 | 4 | 4 | 4 | 1 | 5.2 |
| q_base | 4 | 7 | 5 | 7 | 5 | 4 | 5.6 |
| q_lab | 6 | 8 | 6 | 8 | 6 | 5 | 6.8 |
| q_lab_same | 9 | 5 | 9 | 9 | 7 | 6 | 7.8 |
| q_lab_local | 7 | 6 | 7 | 5 | 8 | 3 | 6.6 |
| r_matte | 10 | 10 | 10 | 10 | 10 | 10 | 10.0 |
| r_medium_harm | 8 | 3 | 8 | 6 | 9 | 7 | 6.8 |

Spearman of `rtdetrv2_real10` with each extractor: resnet50 0.27, dinov2_b14 0.10, siglip2_b16 0.22, dinov3_b16 0.28, clip_b16 0.20. With the mean rank of the 5: 0.31. Mean Spearman among the 5 general ones: 0.82.

### CMMD: rank of each variant (1 = best synthetic quality)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 | rtdetrv2_real10 | mean of the 5 |
|---|---|---|---|---|---|---|---|
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 | 10 | 11.0 |
| g_v11_copypaste | 3 | 3 | 3 | 3 | 2 | 11 | 2.8 |
| j_v11_visible | 2 | 2 | 2 | 2 | 3 | 2 | 2.2 |
| m_pos | 1 | 1 | 1 | 1 | 1 | 5 | 1.0 |
| p_pos_tight | 4 | 9 | 4 | 4 | 4 | 1 | 5.0 |
| q_base | 5 | 7 | 5 | 7 | 5 | 4 | 5.8 |
| q_lab | 6 | 8 | 6 | 8 | 6 | 6 | 6.8 |
| q_lab_same | 9 | 6 | 9 | 9 | 8 | 8 | 8.2 |
| q_lab_local | 8 | 5 | 8 | 5 | 7 | 3 | 6.6 |
| r_matte | 10 | 10 | 10 | 10 | 10 | 9 | 10.0 |
| r_medium_harm | 7 | 4 | 7 | 6 | 9 | 7 | 6.6 |

Spearman of `rtdetrv2_real10` with each extractor: resnet50 0.46, dinov2_b14 0.20, siglip2_b16 0.46, dinov3_b16 0.50, clip_b16 0.41. With the mean rank of the 5: 0.50. Mean Spearman among the 5 general ones: 0.86.

### FD: rank of each variant (1 = best synthetic quality)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 | rtdetrv2_real10 | mean of the 5 |
|---|---|---|---|---|---|---|---|
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 | 9 | 11.0 |
| g_v11_copypaste | 3 | 2 | 2 | 3 | 2 | 11 | 2.4 |
| j_v11_visible | 2 | 3 | 3 | 2 | 3 | 3 | 2.6 |
| m_pos | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |
| p_pos_tight | 8 | 9 | 4 | 5 | 4 | 1 | 6.0 |
| q_base | 7 | 7 | 8 | 9 | 6 | 4 | 7.4 |
| q_lab | 4 | 8 | 5 | 7 | 5 | 7 | 5.8 |
| q_lab_same | 9 | 6 | 9 | 8 | 7 | 6 | 7.8 |
| q_lab_local | 6 | 5 | 6 | 6 | 8 | 2 | 6.2 |
| r_matte | 10 | 10 | 10 | 10 | 10 | 8 | 10.0 |
| r_medium_harm | 5 | 4 | 7 | 4 | 9 | 5 | 5.8 |

Spearman of `rtdetrv2_real10` with each extractor: resnet50 -0.12, dinov2_b14 -0.15, siglip2_b16 -0.04, dinov3_b16 0.02, clip_b16 -0.09. With the mean rank of the 5: -0.15. Mean Spearman among the 5 general ones: 0.84.

## Does it separate v9 from v11?

Separation = AUC(v11) − AUC(v9). A lower AUC means less gap. "Noise" = square root of the sum of squares of the half-widths of the group CIs; "disjoint" = the CIs do not overlap.

| Extractor | AUC v9 (pilot) | AUC p_pos_tight | dAUC / noise / disjoint | AUC px_v9 | AUC px_p | px_p − px_v9: dAUC / noise / disjoint | px_q − px_v9 |
|---|---|---|---|---|---|---|---|
| resnet50 | 0.991 | 0.836 | -0.155 / 0.037 / yes | 0.996 | 0.850 | -0.145 / 0.024 / yes | -0.160 / 0.029 / yes |
| dinov2_b14 | 0.997 | 0.909 | -0.088 / 0.018 / yes | 0.998 | 0.928 | -0.070 / 0.013 / yes | -0.085 / 0.014 / yes |
| siglip2_b16 | 0.999 | 0.921 | -0.078 / 0.019 / yes | 0.999 | 0.928 | -0.071 / 0.013 / yes | -0.084 / 0.018 / yes |
| dinov3_b16 | 0.999 | 0.909 | -0.090 / 0.020 / yes | 1.000 | 0.941 | -0.059 / 0.012 / yes | -0.071 / 0.014 / yes |
| clip_b16 | 1.000 | 0.902 | -0.097 / 0.023 / yes | 1.000 | 0.931 | -0.069 / 0.017 / yes | -0.091 / 0.019 / yes |
| rtdetrv2_real10 | 0.989 | 0.826 | -0.163 / 0.049 / yes | 0.992 | 0.870 | -0.123 / 0.029 / yes | -0.126 / 0.029 / yes |

## Does it discriminate better among the v11 variants and between p and q?

Among the 11 variants (10 of v11 and c_v10_noaug) of the main scenario: pairs of variants with disjoint AUC CIs and dispersion of the AUC across variants divided by the mean CI half-width (the higher, the better it separates above its noise).

| Extractor | Pairs with disjoint CI | Dispersion / CI half-width | Minimum AUC | Maximum AUC |
|---|---|---|---|---|
| resnet50 | 10/55 | 1.06 | 0.794 | 0.937 |
| dinov2_b14 | 15/55 | 1.47 | 0.866 | 0.980 |
| siglip2_b16 | 12/55 | 1.35 | 0.882 | 0.982 |
| dinov3_b16 | 10/55 | 1.24 | 0.895 | 0.992 |
| clip_b16 | 12/55 | 1.33 | 0.852 | 0.987 |
| rtdetrv2_real10 | 11/55 | 0.93 | 0.820 | 0.954 |

p versus q (q − p): transfer-check batches (px_p, px_q) and pilot variants (p_pos_tight, q_lab_same).

| Extractor | transfer check: dAUC / noise / disjoint | pilot: dAUC / noise / disjoint |
|---|---|---|
| resnet50 | -0.015 / 0.037 / no | -0.039 / 0.055 / no |
| dinov2_b14 | -0.015 / 0.019 / no | -0.026 / 0.028 / no |
| siglip2_b16 | -0.013 / 0.023 / no | -0.040 / 0.032 / no |
| dinov3_b16 | -0.012 / 0.018 / no | -0.004 / 0.028 / no |
| clip_b16 | -0.022 / 0.026 / no | -0.039 / 0.042 / no |
| rtdetrv2_real10 | -0.003 / 0.041 / no | -0.006 / 0.070 / no |

## Validation against the AP of the transfer check

Arms: ['px_v9_real10', 'px_p_real10', 'px_q_real10', 'px_v9_real10_s2', 'px_p_real10_s2']; global AP: px_v9_real10 35.00, px_p_real10 36.61, px_q_real10 36.09, px_v9_real10_s2 34.04, px_p_real10_s2 35.15. Concordant pairs (all / with |dAP| ≥ 1) and Spearman with the AP; only 3 batches (exact minimum p 0.33): descriptive.

| Extractor | auc | kid | cmmd | fd | kid@val | cmmd@val | coverage@val | recall@val | tstr_bacc |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| dinov2_b14 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| siglip2_b16 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 3/8 · 2/5 · -0.26 |
| dinov3_b16 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| clip_b16 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| rtdetrv2_real10 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| rtdetrv2_real10_p5 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |
| rtdetrv2_real10_c5 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 | 7/8 · 5/5 · 0.79 |

Order of the batches according to the metric (best → worst), grouped AUC and CMMD:

| Extractor | AUC (arms) | CMMD (arms) |
|---|---|---|
| resnet50 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| dinov2_b14 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 |
| siglip2_b16 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| dinov3_b16 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| clip_b16 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| rtdetrv2_real10 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| rtdetrv2_real10_p5 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |
| rtdetrv2_real10_c5 | px_q_real10 > px_p_real10 > px_p_real10_s2 > px_v9_real10 > px_v9_real10_s2 | px_p_real10 > px_p_real10_s2 > px_q_real10 > px_v9_real10 > px_v9_real10_s2 |

## s_drop vs p_pos_tight (adoption rule)

| Extractor | AUC control [CI] | AUC s_drop [CI] | dAUC | worse | Vendi c / s | Cov@val c / s | low diversity |
|---|---|---|---|---|---|---|---|
| rtdetrv2_real10 | 0.833 [0.795; 0.875] | 0.822 [0.777; 0.867] | -0.011 | no | 2.5 / 2.5 | 0.69 / 0.69 | no |
| rtdetrv2_real10_p5 | 0.823 [0.777; 0.870] | 0.812 [0.760; 0.856] | -0.011 | no | 2.6 / 2.6 | 0.68 / 0.66 | no |
| rtdetrv2_real10_c5 | 0.818 [0.787; 0.867] | 0.818 [0.789; 0.862] | -0.000 | no | 3.7 / 3.7 | 0.65 / 0.67 | no |

- Adoption rule with the 5 general ones: ADOPT; not worse 5/5, diversity not lower 5/5 (minimum 3 in each).
- Adoption rule with the 5 general ones + the task evaluator (6): ADOPT; not worse 6/6, diversity not lower 6/6 (minimum 3 in each).
- Task evaluator only: not worse and diversity not lower.

## Sensitivity to the layer (grouped AUC)

| Variant | rtdetrv2_real10 | rtdetrv2_real10_p5 | rtdetrv2_real10_c5 |
|---|---|---|---|
| _floor | 0.455 | 0.493 | 0.496 |
| ref_batch_v9 | 0.989 | 0.978 | 0.985 |
| c_v10_noaug | 0.954 | 0.932 | 0.945 |
| g_v11_copypaste | 0.896 | 0.891 | 0.874 |
| j_v11_visible | 0.866 | 0.849 | 0.841 |
| m_pos | 0.871 | 0.868 | 0.839 |
| p_pos_tight | 0.826 | 0.812 | 0.814 |
| q_base | 0.844 | 0.803 | 0.828 |
| q_lab | 0.833 | 0.812 | 0.835 |
| q_lab_same | 0.820 | 0.803 | 0.827 |
| q_lab_local | 0.833 | 0.821 | 0.826 |
| r_matte | 0.849 | 0.837 | 0.832 |
| r_medium_harm | 0.826 | 0.807 | 0.818 |

Spearman of ranks (AUC) P3–P5: 0.88; P3–C5: 0.90.

## Seven-class scenario (for Synthetic Seabed)

| Variant | Grouped AUC | KID | CMMD | Coverage@val | TSTR bacc |
|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.440 | 0.0026 | 0.020 | 0.86 | 0.655 |
| ref_batch_v9 | 0.987 | 0.0766 | 0.483 | 0.38 | 0.463 |
| c_v10_noaug | 0.953 | 0.0367 | 0.242 | 0.58 | 0.578 |
| g_v11_copypaste | 0.892 | 0.0416 | 0.232 | 0.63 | 0.585 |
| j_v11_visible | 0.854 | 0.0303 | 0.169 | 0.58 | 0.520 |
| m_pos | 0.869 | 0.0361 | 0.195 | 0.56 | 0.521 |
| p_pos_tight | 0.818 | 0.0220 | 0.129 | 0.69 | 0.575 |
| q_base | 0.840 | 0.0371 | 0.220 | 0.67 | 0.577 |
| q_lab | 0.830 | 0.0415 | 0.243 | 0.69 | 0.512 |
| q_lab_same | 0.823 | 0.0422 | 0.251 | 0.67 | 0.525 |
| q_lab_local | 0.825 | 0.0369 | 0.219 | 0.68 | 0.539 |
| r_matte | 0.832 | 0.0424 | 0.242 | 0.65 | 0.536 |
| r_medium_harm | 0.829 | 0.0413 | 0.240 | 0.65 | 0.527 |
| Synthetic Seabed (7 classes) | 1.000 | 0.1217 | 0.786 | 0.13 | 0.335 |

