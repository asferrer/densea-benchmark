# Real-vs-synthetic gap metrics panel (CPU)

## Summary of the findings (small n; read with the caveats)

Extractors: ResNet-50 (reference), DINOv2 ViT-B/14, SigLIP 2 B/16, DINOv3 ViT-B/16 and CLIP ViT-B/16, all on CPU and on the same crops (quotas of `variant_pilot`). CMMD with sigma = 10 and scale x1000 on L2-normalized embeddings, biased estimator as in the official Google Research code (the unbiased one is also given, `cmmd_unb` in the json). CLIP B/16 is used here and not the L/14@336 of the CMMD paper, so the absolute values are not comparable with the literature.

**1. Does the order of the variants change with the extractor?** It depends on the metric.
- Robust across the 5 extractors: ref_batch_v9 (the worst) and c_v10_noaug are clearly worse than the variants with a real background (copy-paste); Synthetic Seabed has AUC 1.000 with the 5 extractors. The differences among g, j, m_pos, p_pos_tight, q_*, r_* are small.
- KID, CMMD and FD order almost the same across extractors (mean Spearman between extractors 0.86 / 0.89 / 0.88; minimum 0.70 / 0.76 / 0.76). In KID, `m_pos` is the best with the 5 extractors.
- The grouped AUC does change order (mean Spearman 0.63; minimum 0.36) and TSTR more (0.46; minimum 0.27). With DINOv3 the AUC of almost all copy-paste variants tightens to 0.89-0.92. Among the variants g…r the AUC CIs (width of about ±0.03) almost all overlap: **they cannot be ordered with the AUC**.
- The AUC and KID/CMMD disagree: in KID/CMMD q_lab_same and r_matte are worse than m_pos and p_pos_tight with the 5 extractors, whereas the AUC prefers q_lab_same and r_medium_harm to p_pos_tight with the 5 extractors and places m_pos behind q_lab/q_lab_same with SigLIP 2 and CLIP.

**2. Which metric orders the batches like the AP?** None can be declared valid with these data.
- Transfer check (3 batches, 5 arms with the two seeds of v9 and p; 8 pairs): AUC, FD, KID, CMMD, recall, coverage, Vendi (and their versions with VAL reference) agree with the AP in 7 of 8 pairs with the 5 extractors, and in 5 of 5 of the pairs with an AP difference ≥ 1 point. But those 5 pairs are v9 against {p, q}: the gap of v9 is huge (AUC 0.996-1.000 against 0.84-0.94) and every metric separates it. **Within v11, p against q is not discriminated** (p − q = +0.52 AP in seed 1; against the mean of p over two seeds, 35.88, q is +0.21 above; q has a single seed; the noise between seeds is ~1-1.5 AP).
- With n = 3 distinct batches the exact minimum p of Spearman is 0.333: there is no reliable correlation. The 8 pairs are not independent.
- Metrics that disagree with the AP: precision and density with DINOv2 (no agreement in 3/8 pairs), density@val with ResNet-50, nn_real and TSTR with SigLIP 2 (3/8). With so few batches this variation across extractors cannot be attributed to anything.
- Per class (8 classes × batches, up to 64 pairs): the agreement is around 60-75 % for AUC, KID, CMMD, recall, coverage and Vendi (50 % = chance), with differences between metrics (for example 47 against 43 of 64) within the noise of a binomial (standard error ~4 pairs). The per-class TSTR is the worst (~45-65 %). The per-class AP of Metal_Debris (3-6) and Can (10-14) has a noise between seeds comparable to the difference between batches.
- The panel separates clearly worse batches (v9, c_v10_noaug, Seabed: "clearly worse" with every metric and extractor); it does not order similar v11 variants.

**3. Fidelity or coverage?** Mostly coverage.
- With VAL reference, the v11 variants have precision and density close to the real TRAIN-VAL floor with ResNet-50 and SigLIP 2 (reasonable fidelity) and somewhat lower with DINOv2/DINOv3/CLIP, but **coverage@val** is far below the floor with all of them (for example p_pos_tight in the transfer check: 0.31-0.56 depending on the extractor, against a floor of 0.68-0.78). The Vendi of the synthetic sets equals or exceeds the real one: the synthetic set does not lack modes internally, but it does not cover the regions of the real set (shift).
- v9 is worse in both: recall@val 0.09-0.18 (v11: 0.61-0.76) and AUC ≥ 0.996.
- The classifier-based AUC (0.85-0.94) is high although the coverage is partial.

**4. Classes with the largest gaps.** With coverage@val (DINOv2, p_pos_tight): Glove (0.13), Tire (0.28), Plastic_Debris (0.33) and Metal_Debris (0.39) against floors of 0.57-0.96; with q_lab_same Glove improves to 0.22 and Plastic_Debris to 0.46. The TSTR recall is low in Can and Metal_Debris (0.13-0.30 with q_lab_same), classes with few real VAL crops (88 and 23); with SigLIP 2 the highest per-class AUC of p_pos_tight are Fishing_Net (0.99) and Plastic_Debris (0.98). Classes with small n (Metal_Debris with 23 real VAL crops) have very wide intervals.

**5. Methodological caveats.**
- The AUC of the real TRAIN-VAL control with grouped CV is **below 0.5** (0.40-0.48): with disjoint groups between TRAIN and VAL the classifier learns the group identity and fails outside it. The useful floor is ≈0.5; without grouping the control gives 0.63 (ResNet-50, reported in `variant_pilot.md`).
- KID uses the raw features with a polynomial kernel of degree 3 and gamma 1/d: **the scale of KID depends on the extractor** (DINOv2 gives much larger values). Variants are compared within an extractor, not between extractors; the same holds for FD.
- The intervals of the neighbor-based metrics and distances are **stability intervals by group subsampling** (80 %, 30 replicates), not classic CIs; those of the AUC and TSTR are group bootstraps. A single extraction and subsampling seed.
- Precision/recall/density/coverage depend on n (the number of crops per class differs between the floor and the variants); the floor is computed with the same quotas and VAL has fewer crops in Can and Metal_Debris.
- `nn_real` (distance to the closest real item) of the pasted objects is similar to that of the floor and does not indicate near-exact copies: no copying of TRAIN crops is observed, although the pasted objects come from TRAIN crops.
- Seabed has no Mask: the 7-class section compares all the variants with the common minimum quota.

**6. Task evaluator.** Sixth extractor, `rtdetrv2_real10`: RT-DETRv2 R50 of `px_real10/best.pth` (10 % real), P3 map of the HybridEncoder (stride 8, 256 d) averaged inside the box, crop with context at 640x640, CPU. Details and tables in `gap_panel_task.md`.
- It orders the variants like the general extractors only with the AUC (Spearman with the mean rank of the 5: 0.59 against 0.53 among them); with KID, CMMD and FD it separates from them (0.31 / 0.50 / −0.15 against 0.82-0.86 among the general ones).
- It separates v9 from v11 (px_p − px_v9: −0.123, noise 0.029, disjoint CIs) but not better than the others (4.2 noise units against 4.1-6.0).
- It does not discriminate better among the v11 variants (11/55 pairs with disjoint CIs; general ones 10-15) nor between p and q (dAUC −0.003, noise 0.041).
- The verdict of `s_drop` does not change: ADOPT with 6/6 extractors not worse and 6/6 with diversity not lower.
- With 3 batches with AP it agrees like the others (7/8 pairs, 5/5 with |dAP| ≥ 1).


## Extractors and licenses

| Extractor | Weights | License | Dim | Notes |
|---|---|---|---|---|
| resnet50 | torchvision IMAGENET1K_V1 | BSD-3-Clause (torchvision code); weights trained on ImageNet-1k | 2048 | global pooling 2048 d; reference for the AUC of domain_gap |
| dinov2_b14 | facebook/dinov2-base | Apache-2.0 (HF model card) | 768 | ViT-B/14, CLS 768 d, 224 squashed input |
| siglip2_b16 | google/siglip2-base-patch16-224 | Apache-2.0 (HF model card) | 768 | projected image embedding 768 d |
| dinov3_b16 | facebook/dinov3-vitb16-pretrain-lvd1689m | Meta's own license (DINOv3 License; gated repository) | 768 | distilled ViT-B/16, pooler 768 d |
| clip_b16 | openai/clip-vit-base-patch16 | MIT (openai/CLIP repo; the HF model card declares no license) | 512 | projected image embedding 512 d (the one used by CMMD) |

Unavailable: none.

Quotas per class (variant_pilot): {'Bottle': 70, 'Can': 61, 'Fishing_Net': 28, 'Glove': 60, 'Mask': 53, 'Metal_Debris': 59, 'Plastic_Debris': 54, 'Tire': 69}. Quotas of the transfer-check batches: {'Bottle': 125, 'Can': 125, 'Fishing_Net': 125, 'Glove': 125, 'Mask': 125, 'Metal_Debris': 101, 'Plastic_Debris': 125, 'Tire': 125}.

## Order of the variants according to the extractor (1 = best; without Seabed)

Each table orders the variants by the indicated metric with each extractor. The mean and minimum Spearman between pairs of extractors is given (1 = same order). Variants whose differences fall within the CI should not be read as ordered.

### Grouped AUC: mean Spearman between extractors 0.63 (minimum 0.36)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 |
|---|---|---|---|---|---|
| ref_batch_v9 | 12 | 12 | 12 | 12 | 12 |
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 |
| g_v11_copypaste | 10 | 4 | 10 | 2 | 10 |
| j_v11_visible | 5 | 1 | 5 | 1 | 5 |
| m_pos | 3 | 2 | 9 | 4 | 9 |
| p_pos_tight | 8 | 9 | 8 | 6 | 7 |
| q_base | 7 | 8 | 7 | 10 | 6 |
| q_lab | 4 | 6 | 3 | 8 | 4 |
| q_lab_same | 2 | 5 | 1 | 3 | 2 |
| q_lab_local | 6 | 7 | 4 | 5 | 3 |
| r_matte | 9 | 10 | 6 | 9 | 8 |
| r_medium_harm | 1 | 3 | 2 | 7 | 1 |

### KID: mean Spearman between extractors 0.86 (minimum 0.70)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 |
|---|---|---|---|---|---|
| ref_batch_v9 | 12 | 12 | 12 | 12 | 12 |
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 |
| g_v11_copypaste | 3 | 4 | 2 | 3 | 2 |
| j_v11_visible | 2 | 1 | 3 | 2 | 3 |
| m_pos | 1 | 2 | 1 | 1 | 1 |
| p_pos_tight | 5 | 9 | 4 | 4 | 4 |
| q_base | 4 | 7 | 5 | 7 | 5 |
| q_lab | 6 | 8 | 6 | 8 | 6 |
| q_lab_same | 9 | 5 | 9 | 9 | 7 |
| q_lab_local | 7 | 6 | 7 | 5 | 8 |
| r_matte | 10 | 10 | 10 | 10 | 10 |
| r_medium_harm | 8 | 3 | 8 | 6 | 9 |

### CMMD: mean Spearman between extractors 0.89 (minimum 0.76)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 |
|---|---|---|---|---|---|
| ref_batch_v9 | 12 | 12 | 12 | 12 | 12 |
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 |
| g_v11_copypaste | 3 | 3 | 3 | 3 | 2 |
| j_v11_visible | 2 | 2 | 2 | 2 | 3 |
| m_pos | 1 | 1 | 1 | 1 | 1 |
| p_pos_tight | 4 | 9 | 4 | 4 | 4 |
| q_base | 5 | 7 | 5 | 7 | 5 |
| q_lab | 6 | 8 | 6 | 8 | 6 |
| q_lab_same | 9 | 6 | 9 | 9 | 8 |
| q_lab_local | 8 | 5 | 8 | 5 | 7 |
| r_matte | 10 | 10 | 10 | 10 | 10 |
| r_medium_harm | 7 | 4 | 7 | 6 | 9 |

### FD: mean Spearman between extractors 0.88 (minimum 0.76)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 |
|---|---|---|---|---|---|
| ref_batch_v9 | 12 | 12 | 12 | 12 | 12 |
| c_v10_noaug | 11 | 11 | 11 | 11 | 11 |
| g_v11_copypaste | 3 | 2 | 2 | 3 | 2 |
| j_v11_visible | 2 | 3 | 3 | 2 | 3 |
| m_pos | 1 | 1 | 1 | 1 | 1 |
| p_pos_tight | 8 | 9 | 4 | 5 | 4 |
| q_base | 7 | 7 | 8 | 9 | 6 |
| q_lab | 4 | 8 | 5 | 7 | 5 |
| q_lab_same | 9 | 6 | 9 | 8 | 7 |
| q_lab_local | 6 | 5 | 6 | 6 | 8 |
| r_matte | 10 | 10 | 10 | 10 | 10 |
| r_medium_harm | 5 | 4 | 7 | 4 | 9 |

### TSTR (bacc): mean Spearman between extractors 0.46 (minimum 0.27)

| Variant | resnet50 | dinov2_b14 | siglip2_b16 | dinov3_b16 | clip_b16 |
|---|---|---|---|---|---|
| ref_batch_v9 | 12 | 10 | 11 | 9 | 12 |
| c_v10_noaug | 1 | 4 | 8 | 2 | 5 |
| g_v11_copypaste | 3 | 9 | 2 | 10 | 6 |
| j_v11_visible | 5 | 8 | 10 | 12 | 11 |
| m_pos | 10 | 11 | 6 | 8 | 9 |
| p_pos_tight | 4 | 1 | 1 | 3 | 3 |
| q_base | 8 | 2 | 3 | 1 | 10 |
| q_lab | 7 | 6 | 9 | 5 | 1 |
| q_lab_same | 6 | 3 | 5 | 4 | 4 |
| q_lab_local | 9 | 5 | 4 | 7 | 2 |
| r_matte | 11 | 12 | 12 | 11 | 7 |
| r_medium_harm | 2 | 7 | 7 | 6 | 8 |

## Panel per extractor (real reference = TRAIN, and VAL separately)

### resnet50: TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.405 ± 0.010 [0.345; 0.464] | 79.8 | 0.0027 [0.0025; 0.0069] | 0.039 | 0.94 [0.91; 0.96] | 0.82 [0.76; 0.85] | 1.04 [0.85; 1.17] | 0.89 [0.78; 0.92] | 5.6 | 0.158 | 0.639 [0.56; 0.73] |
| ref_batch_v9 | 454 | 0.991 ± 0.001 [0.988; 0.996] | 170.1 | 0.1299 [0.1202; 0.1367] | 0.997 | 0.83 [0.78; 0.88] | 0.14 [0.11; 0.16] | 0.69 [0.60; 0.79] | 0.41 [0.38; 0.45] | 4.8 | 0.172 | 0.298 [0.23; 0.34] |
| c_v10_noaug | 454 | 0.937 ± 0.002 [0.921; 0.962] | 121.1 | 0.0625 [0.0581; 0.0683] | 0.461 | 0.89 [0.88; 0.93] | 0.44 [0.39; 0.48] | 0.89 [0.80; 0.97] | 0.66 [0.63; 0.74] | 5.4 | 0.189 | 0.433 [0.37; 0.52] |
| g_v11_copypaste | 454 | 0.839 ± 0.003 [0.818; 0.878] | 94.1 | 0.0206 [0.0180; 0.0268] | 0.161 | 0.87 [0.83; 0.90] | 0.70 [0.66; 0.73] | 0.88 [0.80; 1.01] | 0.75 [0.70; 0.84] | 6.5 | 0.208 | 0.413 [0.33; 0.49] |
| j_v11_visible | 454 | 0.811 ± 0.002 [0.785; 0.850] | 92.3 | 0.0191 [0.0160; 0.0240] | 0.148 | 0.87 [0.86; 0.91] | 0.68 [0.64; 0.74] | 1.03 [0.96; 1.14] | 0.76 [0.73; 0.83] | 6.5 | 0.209 | 0.407 [0.32; 0.47] |
| m_pos | 454 | 0.809 ± 0.006 [0.778; 0.852] | 88.4 | 0.0133 [0.0102; 0.0174] | 0.114 | 0.91 [0.88; 0.92] | 0.75 [0.67; 0.78] | 1.01 [0.90; 1.06] | 0.75 [0.70; 0.84] | 6.5 | 0.212 | 0.362 [0.31; 0.45] |
| p_pos_tight | 454 | 0.836 ± 0.003 [0.808; 0.881] | 95.3 | 0.0232 [0.0196; 0.0284] | 0.191 | 0.92 [0.90; 0.95] | 0.67 [0.62; 0.71] | 1.02 [0.93; 1.08] | 0.77 [0.72; 0.81] | 6.2 | 0.200 | 0.408 [0.34; 0.48] |
| q_base | 454 | 0.821 ± 0.002 [0.790; 0.867] | 95.0 | 0.0228 [0.0192; 0.0279] | 0.193 | 0.92 [0.90; 0.94] | 0.67 [0.64; 0.71] | 0.93 [0.89; 1.03] | 0.76 [0.73; 0.79] | 6.4 | 0.199 | 0.403 [0.33; 0.51] |
| q_lab | 454 | 0.810 ± 0.005 [0.784; 0.857] | 94.2 | 0.0246 [0.0205; 0.0326] | 0.202 | 0.92 [0.90; 0.95] | 0.68 [0.65; 0.74] | 1.03 [0.95; 1.11] | 0.79 [0.77; 0.84] | 6.3 | 0.194 | 0.405 [0.33; 0.46] |
| q_lab_same | 454 | 0.796 ± 0.002 [0.759; 0.842] | 96.3 | 0.0265 [0.0220; 0.0332] | 0.216 | 0.93 [0.90; 0.96] | 0.69 [0.65; 0.75] | 1.00 [0.95; 1.12] | 0.76 [0.74; 0.82] | 6.4 | 0.196 | 0.405 [0.32; 0.47] |
| q_lab_local | 454 | 0.813 ± 0.004 [0.781; 0.861] | 94.7 | 0.0253 [0.0210; 0.0324] | 0.205 | 0.93 [0.90; 0.94] | 0.67 [0.64; 0.70] | 1.03 [0.95; 1.14] | 0.79 [0.75; 0.83] | 6.3 | 0.199 | 0.397 [0.34; 0.45] |
| r_matte | 454 | 0.837 ± 0.012 [0.804; 0.882] | 100.9 | 0.0320 [0.0268; 0.0396] | 0.252 | 0.92 [0.90; 0.94] | 0.68 [0.64; 0.71] | 0.97 [0.88; 1.06] | 0.75 [0.71; 0.80] | 6.5 | 0.199 | 0.354 [0.27; 0.41] |
| r_medium_harm | 454 | 0.794 ± 0.008 [0.765; 0.836] | 94.7 | 0.0257 [0.0210; 0.0313] | 0.204 | 0.91 [0.89; 0.93] | 0.69 [0.67; 0.73] | 1.01 [0.94; 1.12] | 0.76 [0.73; 0.82] | 6.3 | 0.197 | 0.432 [0.37; 0.51] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [1.000; 1.000] | 193.4 | 0.1084 [0.1046; 0.1148] | 0.783 | 0.57 [0.57; 0.66] | 0.06 [0.05; 0.09] | 0.35 [0.33; 0.43] | 0.27 [0.25; 0.31] | 4.5 | 0.165 | 0.258 [0.19; 0.31] |

### resnet50: VAL reference (no shared TRAIN crops)

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 79.8 | 0.0027 | 0.039 | 0.82 | 0.94 | 0.64 | 0.76 | 0.190 |
| ref_batch_v9 | 181.3 | 0.1476 | 1.153 | 0.84 | 0.13 | 0.66 | 0.28 | 0.206 |
| c_v10_noaug | 125.3 | 0.0675 | 0.501 | 0.82 | 0.47 | 0.66 | 0.46 | 0.201 |
| g_v11_copypaste | 94.8 | 0.0187 | 0.151 | 0.81 | 0.74 | 0.56 | 0.55 | 0.205 |
| j_v11_visible | 91.5 | 0.0150 | 0.114 | 0.79 | 0.77 | 0.63 | 0.57 | 0.203 |
| m_pos | 93.3 | 0.0141 | 0.116 | 0.79 | 0.74 | 0.62 | 0.56 | 0.207 |
| p_pos_tight | 96.2 | 0.0188 | 0.147 | 0.83 | 0.73 | 0.58 | 0.59 | 0.200 |
| q_base | 94.6 | 0.0175 | 0.142 | 0.84 | 0.68 | 0.58 | 0.56 | 0.199 |
| q_lab | 94.1 | 0.0187 | 0.147 | 0.83 | 0.74 | 0.63 | 0.61 | 0.197 |
| q_lab_same | 95.6 | 0.0202 | 0.157 | 0.84 | 0.76 | 0.63 | 0.63 | 0.198 |
| q_lab_local | 93.8 | 0.0191 | 0.148 | 0.82 | 0.74 | 0.63 | 0.61 | 0.196 |
| r_matte | 99.6 | 0.0243 | 0.181 | 0.81 | 0.70 | 0.61 | 0.60 | 0.199 |
| r_medium_harm | 92.9 | 0.0192 | 0.146 | 0.83 | 0.78 | 0.62 | 0.64 | 0.197 |
| Synthetic Seabed (7 classes) | 211.8 | 0.1321 | 0.970 | 0.64 | 0.04 | 0.27 | 0.18 | 0.242 |

### resnet50: per-class AUC (out-of-fold scores of the global classifier)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.34 | 0.40 | 0.49 | 0.39 | 0.44 | 0.35 | 0.36 | 0.39 |
| ref_batch_v9 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 | 0.99 |
| c_v10_noaug | 0.95 | 0.88 | 0.91 | 0.98 | 0.98 | 0.91 | 0.92 | 0.96 |
| g_v11_copypaste | 0.84 | 0.79 | 0.83 | 0.92 | 0.85 | 0.90 | 0.79 | 0.87 |
| j_v11_visible | 0.81 | 0.80 | 0.84 | 0.84 | 0.85 | 0.85 | 0.76 | 0.83 |
| m_pos | 0.81 | 0.74 | 0.81 | 0.88 | 0.83 | 0.83 | 0.80 | 0.87 |
| p_pos_tight | 0.81 | 0.84 | 0.91 | 0.91 | 0.81 | 0.81 | 0.82 | 0.87 |
| q_base | 0.82 | 0.75 | 0.83 | 0.90 | 0.87 | 0.78 | 0.78 | 0.88 |
| q_lab | 0.81 | 0.88 | 0.80 | 0.82 | 0.81 | 0.86 | 0.75 | 0.86 |
| q_lab_same | 0.78 | 0.84 | 0.78 | 0.83 | 0.81 | 0.85 | 0.72 | 0.85 |
| q_lab_local | 0.84 | 0.80 | 0.79 | 0.84 | 0.79 | 0.80 | 0.81 | 0.89 |
| r_matte | 0.81 | 0.81 | 0.76 | 0.90 | 0.91 | 0.89 | 0.81 | 0.86 |
| r_medium_harm | 0.79 | 0.81 | 0.74 | 0.85 | 0.80 | 0.82 | 0.75 | 0.84 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### resnet50: TSTR recall per class (train on pasted crops, evaluate on real VAL)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.71 | 0.52 | 0.64 | 0.68 | 0.64 | 0.35 | 0.70 | 0.86 |
| ref_batch_v9 | 0.76 | 0.25 | 0.11 | 0.18 | 0.34 | 0.22 | 0.24 | 0.29 |
| c_v10_noaug | 0.49 | 0.38 | 0.25 | 0.32 | 0.34 | 0.48 | 0.46 | 0.75 |
| g_v11_copypaste | 0.31 | 0.26 | 0.50 | 0.38 | 0.49 | 0.39 | 0.28 | 0.68 |
| j_v11_visible | 0.21 | 0.41 | 0.50 | 0.40 | 0.45 | 0.30 | 0.30 | 0.68 |
| m_pos | 0.20 | 0.64 | 0.18 | 0.57 | 0.32 | 0.17 | 0.41 | 0.41 |
| p_pos_tight | 0.36 | 0.36 | 0.39 | 0.62 | 0.26 | 0.22 | 0.37 | 0.68 |
| q_base | 0.36 | 0.33 | 0.46 | 0.45 | 0.28 | 0.26 | 0.39 | 0.70 |
| q_lab | 0.40 | 0.23 | 0.61 | 0.35 | 0.49 | 0.09 | 0.33 | 0.74 |
| q_lab_same | 0.43 | 0.20 | 0.57 | 0.42 | 0.43 | 0.13 | 0.30 | 0.77 |
| q_lab_local | 0.37 | 0.28 | 0.46 | 0.50 | 0.42 | 0.04 | 0.33 | 0.77 |
| r_matte | 0.46 | 0.23 | 0.39 | 0.47 | 0.25 | 0.04 | 0.46 | 0.54 |
| r_medium_harm | 0.39 | 0.28 | 0.54 | 0.52 | 0.49 | 0.17 | 0.33 | 0.74 |
| Synthetic Seabed (7 classes) | 0.61 | 0.05 | 0.11 | 0.42 | n/a | 0.13 | 0.17 | 0.32 |

### resnet50: coverage per class (VAL reference)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.70 | 0.48 | 0.86 | 0.85 | 1.00 | 0.48 | 0.94 | 0.74 |
| ref_batch_v9 | 0.27 | 0.34 | 0.50 | 0.08 | 0.21 | 0.57 | 0.46 | 0.22 |
| c_v10_noaug | 0.46 | 0.46 | 0.61 | 0.27 | 0.38 | 0.61 | 0.70 | 0.35 |
| g_v11_copypaste | 0.43 | 0.44 | 0.86 | 0.33 | 0.62 | 0.57 | 0.74 | 0.33 |
| j_v11_visible | 0.56 | 0.49 | 1.00 | 0.38 | 0.66 | 0.61 | 0.52 | 0.51 |
| m_pos | 0.49 | 0.44 | 0.93 | 0.27 | 0.64 | 0.61 | 0.56 | 0.39 |
| p_pos_tight | 0.50 | 0.44 | 0.93 | 0.30 | 0.66 | 0.52 | 0.67 | 0.54 |
| q_base | 0.66 | 0.54 | 1.00 | 0.30 | 0.64 | 0.57 | 0.59 | 0.52 |
| q_lab | 0.54 | 0.38 | 0.96 | 0.33 | 0.75 | 0.48 | 0.48 | 0.58 |
| q_lab_same | 0.56 | 0.46 | 0.93 | 0.38 | 0.75 | 0.48 | 0.43 | 0.62 |
| q_lab_local | 0.56 | 0.41 | 0.96 | 0.30 | 0.70 | 0.48 | 0.56 | 0.55 |
| r_matte | 0.50 | 0.43 | 1.00 | 0.28 | 0.60 | 0.43 | 0.61 | 0.54 |
| r_medium_harm | 0.57 | 0.51 | 0.93 | 0.35 | 0.68 | 0.57 | 0.37 | 0.61 |
| Synthetic Seabed (7 classes) | 0.26 | 0.15 | 0.21 | 0.10 | n/a | 0.61 | 0.20 | 0.07 |

### dinov2_b14: TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.433 ± 0.005 [0.362; 0.511] | 683.6 | 0.1174 [0.1504; 0.3658] | 0.057 | 0.86 [0.82; 0.90] | 0.80 [0.76; 0.84] | 0.92 [0.85; 1.04] | 0.91 [0.82; 0.92] | 16.7 | 0.313 | 0.772 [0.70; 0.82] |
| ref_batch_v9 | 454 | 0.997 ± 0.000 [0.996; 0.999] | 1433.4 | 8.3693 [8.0770; 8.8456] | 1.901 | 0.76 [0.69; 0.81] | 0.18 [0.17; 0.25] | 0.53 [0.47; 0.67] | 0.35 [0.33; 0.43] | 12.3 | 0.314 | 0.463 [0.36; 0.52] |
| c_v10_noaug | 454 | 0.980 ± 0.000 [0.972; 0.989] | 1257.4 | 6.2010 [5.8418; 6.6121] | 1.509 | 0.75 [0.75; 0.81] | 0.39 [0.32; 0.43] | 0.59 [0.55; 0.74] | 0.45 [0.41; 0.56] | 13.7 | 0.339 | 0.584 [0.49; 0.66] |
| g_v11_copypaste | 454 | 0.880 ± 0.010 [0.860; 0.911] | 842.3 | 1.1419 [1.0533; 1.3633] | 0.418 | 0.81 [0.80; 0.89] | 0.71 [0.67; 0.75] | 0.63 [0.60; 0.84] | 0.68 [0.62; 0.74] | 22.2 | 0.449 | 0.471 [0.38; 0.54] |
| j_v11_visible | 454 | 0.866 ± 0.011 [0.848; 0.898] | 846.6 | 0.9666 [0.8401; 1.1131] | 0.379 | 0.77 [0.75; 0.83] | 0.68 [0.64; 0.71] | 0.66 [0.61; 0.85] | 0.67 [0.63; 0.72] | 22.9 | 0.457 | 0.519 [0.42; 0.57] |
| m_pos | 454 | 0.871 ± 0.005 [0.850; 0.900] | 828.9 | 0.9705 [0.8571; 1.1124] | 0.378 | 0.77 [0.76; 0.86] | 0.72 [0.66; 0.77] | 0.57 [0.55; 0.78] | 0.64 [0.59; 0.80] | 23.3 | 0.465 | 0.462 [0.38; 0.51] |
| p_pos_tight | 454 | 0.909 ± 0.007 [0.897; 0.932] | 884.8 | 1.2292 [1.1547; 1.4237] | 0.518 | 0.74 [0.71; 0.83] | 0.67 [0.61; 0.68] | 0.57 [0.53; 0.77] | 0.57 [0.53; 0.63] | 22.3 | 0.451 | 0.631 [0.56; 0.71] |
| q_base | 454 | 0.901 ± 0.003 [0.886; 0.928] | 876.9 | 1.1575 [1.0620; 1.3330] | 0.491 | 0.78 [0.75; 0.85] | 0.72 [0.68; 0.74] | 0.64 [0.62; 0.88] | 0.61 [0.58; 0.67] | 22.5 | 0.453 | 0.604 [0.53; 0.66] |
| q_lab | 454 | 0.887 ± 0.003 [0.871; 0.914] | 877.1 | 1.1717 [1.0687; 1.4393] | 0.495 | 0.78 [0.75; 0.86] | 0.73 [0.69; 0.76] | 0.61 [0.55; 0.83] | 0.60 [0.54; 0.67] | 22.6 | 0.452 | 0.545 [0.47; 0.61] |
| q_lab_same | 454 | 0.883 ± 0.006 [0.867; 0.911] | 872.0 | 1.1424 [1.0304; 1.4106] | 0.485 | 0.78 [0.75; 0.85] | 0.73 [0.68; 0.78] | 0.61 [0.55; 0.81] | 0.63 [0.57; 0.69] | 22.6 | 0.454 | 0.586 [0.52; 0.67] |
| q_lab_local | 454 | 0.898 ± 0.005 [0.880; 0.924] | 869.6 | 1.1530 [1.0621; 1.3160] | 0.470 | 0.77 [0.75; 0.84] | 0.72 [0.68; 0.75] | 0.60 [0.56; 0.86] | 0.61 [0.58; 0.69] | 22.1 | 0.445 | 0.547 [0.49; 0.60] |
| r_matte | 454 | 0.926 ± 0.006 [0.914; 0.950] | 920.3 | 1.4265 [1.3110; 1.6817] | 0.609 | 0.77 [0.75; 0.85] | 0.64 [0.57; 0.70] | 0.62 [0.58; 0.86] | 0.56 [0.54; 0.61] | 22.1 | 0.447 | 0.451 [0.38; 0.50] |
| r_medium_harm | 454 | 0.876 ± 0.003 [0.856; 0.906] | 865.7 | 1.1239 [1.0099; 1.2634] | 0.470 | 0.81 [0.78; 0.85] | 0.72 [0.70; 0.76] | 0.65 [0.59; 0.84] | 0.63 [0.58; 0.68] | 22.6 | 0.453 | 0.539 [0.47; 0.62] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [1.000; 1.000] | 1554.8 | 5.1692 [5.0741; 5.3269] | 1.669 | 0.33 [0.29; 0.38] | 0.02 [0.01; 0.05] | 0.13 [0.10; 0.17] | 0.14 [0.13; 0.18] | 15.4 | 0.351 | 0.475 [0.39; 0.54] |

### dinov2_b14: VAL reference (no shared TRAIN crops)

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 683.6 | 0.1174 | 0.057 | 0.80 | 0.86 | 0.73 | 0.83 | 0.385 |
| ref_batch_v9 | 1504.8 | 8.7156 | 2.034 | 0.68 | 0.19 | 0.35 | 0.24 | 0.417 |
| c_v10_noaug | 1325.8 | 6.4883 | 1.618 | 0.76 | 0.37 | 0.45 | 0.33 | 0.411 |
| g_v11_copypaste | 886.9 | 1.2041 | 0.416 | 0.74 | 0.75 | 0.48 | 0.56 | 0.441 |
| j_v11_visible | 881.9 | 1.0331 | 0.378 | 0.77 | 0.69 | 0.51 | 0.56 | 0.441 |
| m_pos | 875.2 | 1.0471 | 0.377 | 0.76 | 0.72 | 0.45 | 0.52 | 0.452 |
| p_pos_tight | 922.5 | 1.2720 | 0.498 | 0.66 | 0.61 | 0.43 | 0.45 | 0.454 |
| q_base | 912.8 | 1.1971 | 0.470 | 0.70 | 0.69 | 0.48 | 0.50 | 0.452 |
| q_lab | 909.1 | 1.1910 | 0.467 | 0.74 | 0.70 | 0.48 | 0.51 | 0.450 |
| q_lab_same | 907.2 | 1.1685 | 0.459 | 0.73 | 0.72 | 0.48 | 0.50 | 0.449 |
| q_lab_local | 895.1 | 1.1712 | 0.446 | 0.75 | 0.69 | 0.51 | 0.51 | 0.447 |
| r_matte | 950.1 | 1.4410 | 0.574 | 0.75 | 0.64 | 0.51 | 0.47 | 0.449 |
| r_medium_harm | 899.9 | 1.1421 | 0.444 | 0.72 | 0.74 | 0.48 | 0.49 | 0.447 |
| Synthetic Seabed (7 classes) | 1609.7 | 5.3018 | 1.706 | 0.34 | 0.03 | 0.10 | 0.09 | 0.534 |

### dinov2_b14: per-class AUC (out-of-fold scores of the global classifier)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.38 | 0.48 | 0.49 | 0.33 | 0.33 | 0.56 | 0.49 | 0.48 |
| ref_batch_v9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| c_v10_noaug | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 |
| g_v11_copypaste | 0.86 | 0.88 | 0.88 | 0.96 | 0.93 | 0.84 | 0.83 | 0.90 |
| j_v11_visible | 0.84 | 0.86 | 0.84 | 0.94 | 0.92 | 0.81 | 0.86 | 0.91 |
| m_pos | 0.84 | 0.84 | 0.91 | 0.94 | 0.88 | 0.84 | 0.85 | 0.92 |
| p_pos_tight | 0.88 | 0.93 | 0.92 | 0.96 | 0.93 | 0.89 | 0.88 | 0.93 |
| q_base | 0.86 | 0.93 | 0.93 | 0.96 | 0.90 | 0.87 | 0.90 | 0.93 |
| q_lab | 0.85 | 0.94 | 0.91 | 0.92 | 0.88 | 0.89 | 0.85 | 0.92 |
| q_lab_same | 0.86 | 0.92 | 0.88 | 0.93 | 0.86 | 0.90 | 0.83 | 0.91 |
| q_lab_local | 0.84 | 0.94 | 0.87 | 0.97 | 0.91 | 0.90 | 0.90 | 0.91 |
| r_matte | 0.90 | 0.94 | 0.93 | 0.98 | 0.96 | 0.90 | 0.91 | 0.95 |
| r_medium_harm | 0.83 | 0.92 | 0.82 | 0.92 | 0.85 | 0.88 | 0.83 | 0.94 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### dinov2_b14: TSTR recall per class (train on pasted crops, evaluate on real VAL)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.81 | 0.70 | 0.64 | 0.88 | 0.81 | 0.65 | 0.85 | 0.81 |
| ref_batch_v9 | 0.24 | 0.56 | 0.57 | 0.62 | 0.74 | 0.09 | 0.39 | 0.51 |
| c_v10_noaug | 0.60 | 0.75 | 0.25 | 0.73 | 0.40 | 0.48 | 0.65 | 0.81 |
| g_v11_copypaste | 0.47 | 0.28 | 0.50 | 0.70 | 0.25 | 0.35 | 0.44 | 0.78 |
| j_v11_visible | 0.60 | 0.49 | 0.46 | 0.68 | 0.36 | 0.35 | 0.35 | 0.86 |
| m_pos | 0.39 | 0.77 | 0.32 | 0.32 | 0.38 | 0.17 | 0.61 | 0.74 |
| p_pos_tight | 0.70 | 0.56 | 0.57 | 0.63 | 0.64 | 0.52 | 0.56 | 0.87 |
| q_base | 0.69 | 0.52 | 0.61 | 0.65 | 0.57 | 0.43 | 0.46 | 0.90 |
| q_lab | 0.79 | 0.16 | 0.54 | 0.58 | 0.66 | 0.17 | 0.56 | 0.90 |
| q_lab_same | 0.79 | 0.30 | 0.54 | 0.72 | 0.62 | 0.17 | 0.65 | 0.91 |
| q_lab_local | 0.70 | 0.31 | 0.57 | 0.87 | 0.32 | 0.26 | 0.54 | 0.81 |
| r_matte | 0.73 | 0.13 | 0.50 | 0.28 | 0.11 | 0.22 | 0.83 | 0.80 |
| r_medium_harm | 0.64 | 0.38 | 0.46 | 0.58 | 0.53 | 0.17 | 0.69 | 0.86 |
| Synthetic Seabed (7 classes) | 0.44 | 0.25 | 0.36 | 0.60 | n/a | 0.48 | 0.52 | 0.68 |

### dinov2_b14: coverage per class (VAL reference)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.80 | 0.77 | 0.93 | 0.90 | 0.92 | 0.57 | 0.96 | 0.86 |
| ref_batch_v9 | 0.23 | 0.33 | 0.46 | 0.12 | 0.28 | 0.43 | 0.33 | 0.14 |
| c_v10_noaug | 0.39 | 0.43 | 0.54 | 0.15 | 0.32 | 0.48 | 0.50 | 0.29 |
| g_v11_copypaste | 0.54 | 0.41 | 0.71 | 0.33 | 0.64 | 0.35 | 0.59 | 0.35 |
| j_v11_visible | 0.67 | 0.43 | 0.71 | 0.25 | 0.58 | 0.52 | 0.44 | 0.26 |
| m_pos | 0.47 | 0.61 | 0.75 | 0.12 | 0.58 | 0.30 | 0.44 | 0.39 |
| p_pos_tight | 0.60 | 0.61 | 0.68 | 0.13 | 0.72 | 0.39 | 0.33 | 0.28 |
| q_base | 0.54 | 0.57 | 0.89 | 0.17 | 0.60 | 0.30 | 0.41 | 0.29 |
| q_lab | 0.64 | 0.44 | 0.75 | 0.23 | 0.68 | 0.35 | 0.48 | 0.41 |
| q_lab_same | 0.61 | 0.43 | 0.89 | 0.22 | 0.68 | 0.43 | 0.46 | 0.42 |
| q_lab_local | 0.56 | 0.39 | 0.89 | 0.23 | 0.55 | 0.30 | 0.41 | 0.41 |
| r_matte | 0.53 | 0.49 | 0.68 | 0.13 | 0.45 | 0.35 | 0.52 | 0.28 |
| r_medium_harm | 0.59 | 0.43 | 0.71 | 0.22 | 0.75 | 0.39 | 0.37 | 0.42 |
| Synthetic Seabed (7 classes) | 0.23 | 0.18 | 0.11 | 0.05 | n/a | 0.22 | 0.19 | 0.07 |

### siglip2_b16: TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.411 ± 0.007 [0.347; 0.470] | 9.6 | 0.0011 [0.0008; 0.0044] | 0.027 | 0.86 [0.81; 0.91] | 0.78 [0.75; 0.82] | 1.07 [0.90; 1.19] | 0.91 [0.81; 0.95] | 3.4 | 0.097 | 0.733 [0.66; 0.79] |
| ref_batch_v9 | 454 | 0.999 ± 0.000 [0.997; 0.999] | 29.4 | 0.0620 [0.0577; 0.0653] | 0.916 | 0.57 [0.52; 0.63] | 0.21 [0.18; 0.25] | 0.31 [0.27; 0.37] | 0.31 [0.29; 0.35] | 3.2 | 0.117 | 0.532 [0.47; 0.59] |
| c_v10_noaug | 454 | 0.982 ± 0.001 [0.975; 0.990] | 18.9 | 0.0293 [0.0262; 0.0327] | 0.432 | 0.76 [0.74; 0.83] | 0.41 [0.36; 0.45] | 0.74 [0.65; 0.96] | 0.59 [0.57; 0.69] | 3.4 | 0.117 | 0.558 [0.50; 0.67] |
| g_v11_copypaste | 454 | 0.929 ± 0.002 [0.916; 0.952] | 14.0 | 0.0118 [0.0106; 0.0142] | 0.170 | 0.78 [0.76; 0.83] | 0.54 [0.49; 0.57] | 0.89 [0.84; 1.11] | 0.66 [0.62; 0.75] | 3.4 | 0.116 | 0.605 [0.54; 0.66] |
| j_v11_visible | 454 | 0.912 ± 0.005 [0.896; 0.937] | 14.3 | 0.0118 [0.0100; 0.0133] | 0.161 | 0.81 [0.78; 0.85] | 0.52 [0.46; 0.58] | 1.06 [0.96; 1.23] | 0.67 [0.64; 0.74] | 3.4 | 0.117 | 0.534 [0.46; 0.64] |
| m_pos | 454 | 0.922 ± 0.003 [0.908; 0.943] | 13.3 | 0.0084 [0.0075; 0.0095] | 0.128 | 0.81 [0.76; 0.84] | 0.58 [0.51; 0.62] | 0.90 [0.82; 0.99] | 0.70 [0.66; 0.77] | 3.5 | 0.120 | 0.560 [0.50; 0.65] |
| p_pos_tight | 454 | 0.921 ± 0.003 [0.908; 0.946] | 14.8 | 0.0147 [0.0123; 0.0179] | 0.205 | 0.81 [0.79; 0.85] | 0.57 [0.53; 0.63] | 0.97 [0.86; 1.11] | 0.69 [0.65; 0.74] | 3.3 | 0.112 | 0.609 [0.54; 0.69] |
| q_base | 454 | 0.919 ± 0.007 [0.902; 0.943] | 15.6 | 0.0173 [0.0147; 0.0203] | 0.225 | 0.83 [0.80; 0.86] | 0.54 [0.48; 0.58] | 0.96 [0.92; 1.17] | 0.64 [0.62; 0.71] | 3.2 | 0.107 | 0.590 [0.53; 0.68] |
| q_lab | 454 | 0.886 ± 0.001 [0.872; 0.916] | 15.3 | 0.0179 [0.0147; 0.0222] | 0.230 | 0.87 [0.85; 0.91] | 0.60 [0.55; 0.66] | 1.07 [0.95; 1.27] | 0.74 [0.72; 0.78] | 3.2 | 0.104 | 0.558 [0.49; 0.65] |
| q_lab_same | 454 | 0.882 ± 0.003 [0.864; 0.917] | 15.9 | 0.0188 [0.0158; 0.0238] | 0.241 | 0.87 [0.84; 0.90] | 0.61 [0.57; 0.66] | 1.05 [0.96; 1.23] | 0.69 [0.66; 0.74] | 3.2 | 0.104 | 0.564 [0.50; 0.63] |
| q_lab_local | 454 | 0.892 ± 0.005 [0.874; 0.925] | 15.5 | 0.0183 [0.0154; 0.0234] | 0.235 | 0.88 [0.85; 0.91] | 0.56 [0.53; 0.63] | 1.18 [1.08; 1.37] | 0.71 [0.68; 0.76] | 3.2 | 0.104 | 0.589 [0.53; 0.67] |
| r_matte | 454 | 0.912 ± 0.004 [0.897; 0.938] | 16.8 | 0.0233 [0.0202; 0.0276] | 0.281 | 0.87 [0.86; 0.90] | 0.52 [0.48; 0.58] | 1.22 [1.13; 1.41] | 0.71 [0.67; 0.77] | 3.1 | 0.101 | 0.489 [0.42; 0.54] |
| r_medium_harm | 454 | 0.883 ± 0.003 [0.862; 0.913] | 15.5 | 0.0183 [0.0149; 0.0206] | 0.234 | 0.85 [0.83; 0.88] | 0.61 [0.58; 0.66] | 1.04 [0.94; 1.19] | 0.69 [0.66; 0.73] | 3.2 | 0.104 | 0.558 [0.49; 0.66] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [1.000; 1.000] | 36.4 | 0.0678 [0.0664; 0.0708] | 0.987 | 0.20 [0.18; 0.27] | 0.05 [0.02; 0.06] | 0.07 [0.06; 0.10] | 0.07 [0.06; 0.11] | 2.8 | 0.097 | 0.368 [0.30; 0.45] |

### siglip2_b16: VAL reference (no shared TRAIN crops)

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 9.6 | 0.0011 | 0.027 | 0.78 | 0.86 | 0.70 | 0.85 | 0.109 |
| ref_batch_v9 | 31.7 | 0.0712 | 1.039 | 0.57 | 0.17 | 0.31 | 0.26 | 0.154 |
| c_v10_noaug | 19.9 | 0.0341 | 0.500 | 0.74 | 0.47 | 0.75 | 0.54 | 0.132 |
| g_v11_copypaste | 13.3 | 0.0095 | 0.142 | 0.75 | 0.60 | 0.70 | 0.60 | 0.119 |
| j_v11_visible | 13.1 | 0.0082 | 0.118 | 0.76 | 0.58 | 0.83 | 0.65 | 0.115 |
| m_pos | 12.6 | 0.0065 | 0.104 | 0.76 | 0.63 | 0.71 | 0.63 | 0.122 |
| p_pos_tight | 13.3 | 0.0101 | 0.147 | 0.73 | 0.60 | 0.71 | 0.65 | 0.113 |
| q_base | 14.0 | 0.0122 | 0.164 | 0.73 | 0.58 | 0.76 | 0.61 | 0.109 |
| q_lab | 13.8 | 0.0123 | 0.164 | 0.79 | 0.62 | 0.75 | 0.66 | 0.106 |
| q_lab_same | 14.2 | 0.0131 | 0.172 | 0.78 | 0.62 | 0.73 | 0.63 | 0.105 |
| q_lab_local | 13.7 | 0.0125 | 0.166 | 0.81 | 0.61 | 0.85 | 0.68 | 0.105 |
| r_matte | 15.0 | 0.0167 | 0.205 | 0.83 | 0.59 | 0.89 | 0.67 | 0.101 |
| r_medium_harm | 13.7 | 0.0124 | 0.163 | 0.77 | 0.61 | 0.77 | 0.65 | 0.104 |
| Synthetic Seabed (7 classes) | 37.8 | 0.0721 | 1.047 | 0.14 | 0.07 | 0.03 | 0.05 | 0.179 |

### siglip2_b16: per-class AUC (out-of-fold scores of the global classifier)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.29 | 0.51 | 0.50 | 0.36 | 0.31 | 0.60 | 0.39 | 0.37 |
| ref_batch_v9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| c_v10_noaug | 0.98 | 0.98 | 0.99 | 0.99 | 0.97 | 0.95 | 0.99 | 0.99 |
| g_v11_copypaste | 0.91 | 0.91 | 0.91 | 0.98 | 0.92 | 0.91 | 0.95 | 0.96 |
| j_v11_visible | 0.85 | 0.91 | 0.96 | 0.95 | 0.90 | 0.89 | 0.95 | 0.95 |
| m_pos | 0.90 | 0.93 | 0.99 | 0.96 | 0.89 | 0.92 | 0.92 | 0.94 |
| p_pos_tight | 0.87 | 0.88 | 0.99 | 0.95 | 0.92 | 0.92 | 0.98 | 0.95 |
| q_base | 0.88 | 0.87 | 0.98 | 0.92 | 0.92 | 0.91 | 0.96 | 0.97 |
| q_lab | 0.83 | 0.91 | 0.97 | 0.90 | 0.86 | 0.89 | 0.89 | 0.95 |
| q_lab_same | 0.83 | 0.89 | 0.96 | 0.90 | 0.83 | 0.89 | 0.90 | 0.95 |
| q_lab_local | 0.87 | 0.89 | 0.92 | 0.94 | 0.88 | 0.85 | 0.90 | 0.95 |
| r_matte | 0.89 | 0.90 | 0.94 | 0.96 | 0.93 | 0.91 | 0.91 | 0.95 |
| r_medium_harm | 0.83 | 0.89 | 0.99 | 0.92 | 0.86 | 0.85 | 0.90 | 0.94 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### siglip2_b16: TSTR recall per class (train on pasted crops, evaluate on real VAL)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.77 | 0.56 | 0.79 | 0.88 | 0.74 | 0.43 | 0.80 | 0.90 |
| ref_batch_v9 | 0.34 | 0.57 | 0.71 | 0.85 | 0.68 | 0.26 | 0.50 | 0.33 |
| c_v10_noaug | 0.54 | 0.59 | 0.50 | 0.80 | 0.49 | 0.30 | 0.74 | 0.49 |
| g_v11_copypaste | 0.54 | 0.46 | 0.82 | 0.83 | 0.36 | 0.48 | 0.67 | 0.68 |
| j_v11_visible | 0.67 | 0.26 | 0.61 | 0.75 | 0.57 | 0.43 | 0.43 | 0.55 |
| m_pos | 0.44 | 0.61 | 0.61 | 0.83 | 0.55 | 0.26 | 0.57 | 0.61 |
| p_pos_tight | 0.60 | 0.38 | 0.71 | 0.92 | 0.55 | 0.52 | 0.43 | 0.77 |
| q_base | 0.64 | 0.46 | 0.89 | 0.73 | 0.57 | 0.17 | 0.61 | 0.64 |
| q_lab | 0.69 | 0.20 | 0.71 | 0.73 | 0.51 | 0.22 | 0.72 | 0.68 |
| q_lab_same | 0.67 | 0.23 | 0.75 | 0.78 | 0.60 | 0.13 | 0.63 | 0.71 |
| q_lab_local | 0.66 | 0.23 | 0.79 | 0.90 | 0.49 | 0.13 | 0.70 | 0.81 |
| r_matte | 0.53 | 0.15 | 0.75 | 0.68 | 0.55 | 0.04 | 0.59 | 0.62 |
| r_medium_harm | 0.56 | 0.25 | 0.68 | 0.78 | 0.53 | 0.30 | 0.70 | 0.67 |
| Synthetic Seabed (7 classes) | 0.50 | 0.23 | 0.54 | 0.53 | n/a | 0.17 | 0.30 | 0.30 |

### siglip2_b16: coverage per class (VAL reference)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.69 | 0.70 | 0.86 | 0.95 | 0.94 | 0.57 | 0.98 | 0.83 |
| ref_batch_v9 | 0.17 | 0.30 | 0.50 | 0.25 | 0.38 | 0.35 | 0.39 | 0.14 |
| c_v10_noaug | 0.47 | 0.67 | 0.46 | 0.37 | 0.45 | 0.52 | 0.52 | 0.45 |
| g_v11_copypaste | 0.57 | 0.62 | 0.75 | 0.40 | 0.66 | 0.52 | 0.61 | 0.55 |
| j_v11_visible | 0.51 | 0.62 | 0.71 | 0.42 | 0.64 | 0.52 | 0.50 | 0.57 |
| m_pos | 0.60 | 0.74 | 0.79 | 0.42 | 0.74 | 0.52 | 0.57 | 0.49 |
| p_pos_tight | 0.63 | 0.64 | 0.71 | 0.40 | 0.70 | 0.52 | 0.59 | 0.61 |
| q_base | 0.53 | 0.66 | 0.82 | 0.37 | 0.70 | 0.52 | 0.67 | 0.61 |
| q_lab | 0.61 | 0.52 | 0.79 | 0.35 | 0.79 | 0.57 | 0.69 | 0.68 |
| q_lab_same | 0.53 | 0.54 | 0.64 | 0.33 | 0.75 | 0.57 | 0.61 | 0.64 |
| q_lab_local | 0.61 | 0.54 | 0.82 | 0.53 | 0.81 | 0.57 | 0.50 | 0.72 |
| r_matte | 0.66 | 0.56 | 0.82 | 0.45 | 0.66 | 0.43 | 0.69 | 0.65 |
| r_medium_harm | 0.56 | 0.59 | 0.79 | 0.38 | 0.75 | 0.52 | 0.59 | 0.64 |
| Synthetic Seabed (7 classes) | 0.10 | 0.07 | 0.21 | 0.08 | n/a | 0.35 | 0.13 | 0.01 |

### dinov3_b16: TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.432 ± 0.026 [0.376; 0.493] | 65.4 | 0.0028 [0.0037; 0.0114] | 0.062 | 0.90 [0.84; 0.92] | 0.81 [0.76; 0.85] | 0.94 [0.83; 1.03] | 0.84 [0.75; 0.88] | 17.3 | 0.316 | 0.778 [0.72; 0.84] |
| ref_batch_v9 | 454 | 0.999 ± 0.000 [0.998; 1.000] | 133.6 | 0.1971 [0.1881; 0.2104] | 1.763 | 0.66 [0.65; 0.74] | 0.10 [0.08; 0.13] | 0.37 [0.35; 0.47] | 0.31 [0.29; 0.39] | 15.4 | 0.379 | 0.530 [0.46; 0.60] |
| c_v10_noaug | 454 | 0.992 ± 0.001 [0.986; 0.996] | 115.6 | 0.1372 [0.1287; 0.1459] | 1.252 | 0.65 [0.60; 0.76] | 0.40 [0.33; 0.43] | 0.38 [0.32; 0.47] | 0.39 [0.37; 0.48] | 18.8 | 0.419 | 0.616 [0.52; 0.68] |
| g_v11_copypaste | 454 | 0.901 ± 0.003 [0.880; 0.933] | 81.0 | 0.0381 [0.0357; 0.0449] | 0.429 | 0.80 [0.77; 0.85] | 0.68 [0.61; 0.72] | 0.58 [0.52; 0.76] | 0.61 [0.56; 0.67] | 24.9 | 0.484 | 0.518 [0.46; 0.60] |
| j_v11_visible | 454 | 0.895 ± 0.007 [0.873; 0.923] | 80.1 | 0.0328 [0.0291; 0.0369] | 0.379 | 0.76 [0.74; 0.84] | 0.72 [0.68; 0.75] | 0.54 [0.51; 0.71] | 0.60 [0.56; 0.66] | 26.5 | 0.504 | 0.500 [0.42; 0.58] |
| m_pos | 454 | 0.906 ± 0.001 [0.892; 0.931] | 78.3 | 0.0308 [0.0277; 0.0339] | 0.358 | 0.73 [0.71; 0.81] | 0.79 [0.75; 0.82] | 0.49 [0.45; 0.61] | 0.57 [0.54; 0.71] | 26.7 | 0.505 | 0.543 [0.48; 0.62] |
| p_pos_tight | 454 | 0.909 ± 0.006 [0.896; 0.936] | 82.1 | 0.0404 [0.0381; 0.0471] | 0.461 | 0.76 [0.74; 0.82] | 0.75 [0.69; 0.79] | 0.50 [0.45; 0.62] | 0.56 [0.52; 0.62] | 24.9 | 0.485 | 0.616 [0.55; 0.71] |
| q_base | 454 | 0.920 ± 0.006 [0.903; 0.948] | 83.5 | 0.0417 [0.0381; 0.0482] | 0.476 | 0.77 [0.73; 0.82] | 0.74 [0.69; 0.79] | 0.50 [0.45; 0.63] | 0.53 [0.50; 0.58] | 25.4 | 0.485 | 0.633 [0.58; 0.70] |
| q_lab | 454 | 0.911 ± 0.003 [0.897; 0.937] | 82.5 | 0.0418 [0.0381; 0.0515] | 0.478 | 0.75 [0.72; 0.85] | 0.76 [0.71; 0.79] | 0.51 [0.47; 0.69] | 0.56 [0.52; 0.63] | 25.1 | 0.478 | 0.588 [0.53; 0.66] |
| q_lab_same | 454 | 0.905 ± 0.002 [0.890; 0.931] | 83.4 | 0.0426 [0.0383; 0.0522] | 0.486 | 0.75 [0.72; 0.86] | 0.75 [0.70; 0.79] | 0.51 [0.46; 0.69] | 0.54 [0.50; 0.60] | 25.2 | 0.480 | 0.593 [0.53; 0.68] |
| q_lab_local | 454 | 0.909 ± 0.004 [0.893; 0.940] | 82.2 | 0.0405 [0.0380; 0.0473] | 0.462 | 0.78 [0.76; 0.86] | 0.70 [0.65; 0.76] | 0.52 [0.49; 0.70] | 0.55 [0.53; 0.64] | 25.0 | 0.481 | 0.548 [0.49; 0.63] |
| r_matte | 454 | 0.916 ± 0.004 [0.907; 0.940] | 86.5 | 0.0489 [0.0441; 0.0579] | 0.555 | 0.77 [0.74; 0.85] | 0.62 [0.58; 0.67] | 0.54 [0.49; 0.68] | 0.53 [0.51; 0.62] | 25.1 | 0.480 | 0.515 [0.44; 0.59] |
| r_medium_harm | 454 | 0.910 ± 0.001 [0.893; 0.937] | 81.8 | 0.0409 [0.0369; 0.0466] | 0.469 | 0.81 [0.77; 0.88] | 0.75 [0.73; 0.79] | 0.57 [0.49; 0.73] | 0.57 [0.54; 0.63] | 25.1 | 0.482 | 0.571 [0.52; 0.65] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [1.000; 1.000] | 156.1 | 0.1978 [0.1935; 0.2017] | 2.126 | 0.23 [0.17; 0.28] | 0.00 [0.00; 0.01] | 0.07 [0.05; 0.09] | 0.09 [0.08; 0.15] | 15.2 | 0.359 | 0.463 [0.40; 0.53] |

### dinov3_b16: VAL reference (no shared TRAIN crops)

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 65.4 | 0.0028 | 0.062 | 0.81 | 0.90 | 0.67 | 0.79 | 0.411 |
| ref_batch_v9 | 142.2 | 0.2093 | 1.897 | 0.63 | 0.13 | 0.26 | 0.21 | 0.495 |
| c_v10_noaug | 123.1 | 0.1450 | 1.340 | 0.63 | 0.39 | 0.29 | 0.28 | 0.491 |
| g_v11_copypaste | 87.5 | 0.0417 | 0.467 | 0.69 | 0.69 | 0.37 | 0.47 | 0.489 |
| j_v11_visible | 85.3 | 0.0350 | 0.401 | 0.69 | 0.71 | 0.37 | 0.47 | 0.495 |
| m_pos | 86.2 | 0.0340 | 0.390 | 0.64 | 0.77 | 0.33 | 0.43 | 0.503 |
| p_pos_tight | 87.9 | 0.0418 | 0.472 | 0.67 | 0.71 | 0.34 | 0.42 | 0.499 |
| q_base | 88.2 | 0.0426 | 0.482 | 0.65 | 0.68 | 0.33 | 0.41 | 0.500 |
| q_lab | 87.1 | 0.0417 | 0.474 | 0.70 | 0.75 | 0.38 | 0.40 | 0.497 |
| q_lab_same | 88.1 | 0.0423 | 0.479 | 0.70 | 0.75 | 0.36 | 0.39 | 0.496 |
| q_lab_local | 86.5 | 0.0403 | 0.457 | 0.73 | 0.70 | 0.38 | 0.41 | 0.492 |
| r_matte | 90.8 | 0.0480 | 0.541 | 0.70 | 0.62 | 0.40 | 0.40 | 0.493 |
| r_medium_harm | 86.5 | 0.0404 | 0.460 | 0.67 | 0.74 | 0.37 | 0.41 | 0.492 |
| Synthetic Seabed (7 classes) | 165.4 | 0.2059 | 2.211 | 0.16 | 0.01 | 0.04 | 0.07 | 0.588 |

### dinov3_b16: per-class AUC (out-of-fold scores of the global classifier)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.44 | 0.46 | 0.56 | 0.41 | 0.39 | 0.26 | 0.44 | 0.40 |
| ref_batch_v9 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| c_v10_noaug | 1.00 | 0.98 | 0.96 | 1.00 | 0.99 | 0.99 | 0.98 | 1.00 |
| g_v11_copypaste | 0.90 | 0.89 | 0.84 | 0.96 | 0.91 | 0.89 | 0.88 | 0.94 |
| j_v11_visible | 0.86 | 0.88 | 0.85 | 0.95 | 0.94 | 0.85 | 0.87 | 0.95 |
| m_pos | 0.88 | 0.83 | 0.89 | 0.96 | 0.90 | 0.93 | 0.92 | 0.95 |
| p_pos_tight | 0.89 | 0.90 | 0.93 | 0.96 | 0.88 | 0.91 | 0.91 | 0.94 |
| q_base | 0.91 | 0.87 | 0.94 | 0.96 | 0.91 | 0.92 | 0.91 | 0.96 |
| q_lab | 0.86 | 0.92 | 0.93 | 0.94 | 0.88 | 0.96 | 0.91 | 0.96 |
| q_lab_same | 0.87 | 0.96 | 0.81 | 0.94 | 0.86 | 0.94 | 0.92 | 0.95 |
| q_lab_local | 0.90 | 0.91 | 0.85 | 0.96 | 0.90 | 0.91 | 0.92 | 0.94 |
| r_matte | 0.88 | 0.91 | 0.86 | 0.97 | 0.91 | 0.94 | 0.93 | 0.96 |
| r_medium_harm | 0.89 | 0.93 | 0.85 | 0.95 | 0.85 | 0.94 | 0.89 | 0.96 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### dinov3_b16: TSTR recall per class (train on pasted crops, evaluate on real VAL)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.84 | 0.62 | 0.86 | 0.90 | 0.92 | 0.35 | 0.89 | 0.84 |
| ref_batch_v9 | 0.43 | 0.48 | 0.64 | 0.83 | 0.77 | 0.22 | 0.33 | 0.54 |
| c_v10_noaug | 0.60 | 0.80 | 0.46 | 0.73 | 0.38 | 0.57 | 0.70 | 0.68 |
| g_v11_copypaste | 0.46 | 0.31 | 0.61 | 0.73 | 0.49 | 0.22 | 0.52 | 0.81 |
| j_v11_visible | 0.56 | 0.30 | 0.61 | 0.82 | 0.58 | 0.17 | 0.37 | 0.59 |
| m_pos | 0.50 | 0.67 | 0.43 | 0.52 | 0.53 | 0.30 | 0.80 | 0.59 |
| p_pos_tight | 0.60 | 0.52 | 0.68 | 0.67 | 0.75 | 0.17 | 0.70 | 0.83 |
| q_base | 0.70 | 0.62 | 0.82 | 0.73 | 0.53 | 0.09 | 0.76 | 0.81 |
| q_lab | 0.73 | 0.26 | 0.71 | 0.70 | 0.66 | 0.17 | 0.59 | 0.87 |
| q_lab_same | 0.73 | 0.31 | 0.71 | 0.63 | 0.62 | 0.17 | 0.65 | 0.91 |
| q_lab_local | 0.50 | 0.31 | 0.64 | 0.78 | 0.49 | 0.17 | 0.67 | 0.81 |
| r_matte | 0.71 | 0.20 | 0.64 | 0.38 | 0.38 | 0.13 | 0.83 | 0.84 |
| r_medium_harm | 0.57 | 0.36 | 0.68 | 0.58 | 0.55 | 0.13 | 0.81 | 0.88 |
| Synthetic Seabed (7 classes) | 0.53 | 0.25 | 0.32 | 0.50 | n/a | 0.65 | 0.44 | 0.55 |

### dinov3_b16: coverage per class (VAL reference)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.71 | 0.66 | 0.96 | 0.78 | 0.96 | 0.48 | 0.93 | 0.78 |
| ref_batch_v9 | 0.26 | 0.23 | 0.43 | 0.12 | 0.26 | 0.35 | 0.28 | 0.16 |
| c_v10_noaug | 0.37 | 0.25 | 0.50 | 0.17 | 0.36 | 0.35 | 0.48 | 0.17 |
| g_v11_copypaste | 0.44 | 0.28 | 0.54 | 0.25 | 0.53 | 0.30 | 0.72 | 0.17 |
| j_v11_visible | 0.59 | 0.39 | 0.71 | 0.20 | 0.42 | 0.35 | 0.28 | 0.25 |
| m_pos | 0.39 | 0.44 | 0.71 | 0.12 | 0.49 | 0.22 | 0.37 | 0.23 |
| p_pos_tight | 0.46 | 0.41 | 0.68 | 0.13 | 0.57 | 0.26 | 0.39 | 0.32 |
| q_base | 0.39 | 0.38 | 0.57 | 0.12 | 0.55 | 0.30 | 0.33 | 0.41 |
| q_lab | 0.44 | 0.36 | 0.64 | 0.17 | 0.57 | 0.26 | 0.43 | 0.29 |
| q_lab_same | 0.47 | 0.28 | 0.68 | 0.15 | 0.58 | 0.26 | 0.39 | 0.29 |
| q_lab_local | 0.40 | 0.36 | 0.61 | 0.13 | 0.51 | 0.26 | 0.44 | 0.29 |
| r_matte | 0.57 | 0.36 | 0.57 | 0.12 | 0.49 | 0.26 | 0.43 | 0.30 |
| r_medium_harm | 0.50 | 0.31 | 0.75 | 0.17 | 0.60 | 0.26 | 0.35 | 0.30 |
| Synthetic Seabed (7 classes) | 0.19 | 0.05 | 0.14 | 0.05 | n/a | 0.17 | 0.17 | 0.03 |

### clip_b16: TRAIN reference

| Variant | n | Grouped AUC ± sd [group CI] | FD | KID | CMMD | Precision | Recall | Density | Coverage | Vendi | NN dist | TSTR bacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 418 | 0.412 ± 0.006 [0.341; 0.467] | 7.0 | 0.0026 [0.0014; 0.0061] | 0.036 | 0.85 [0.82; 0.90] | 0.80 [0.76; 0.84] | 1.11 [1.01; 1.18] | 0.94 [0.84; 0.96] | 3.3 | 0.098 | 0.697 [0.64; 0.78] |
| ref_batch_v9 | 454 | 1.000 ± 0.000 [0.999; 1.000] | 23.7 | 0.0901 [0.0833; 0.0957] | 0.929 | 0.55 [0.51; 0.61] | 0.14 [0.10; 0.17] | 0.32 [0.28; 0.37] | 0.32 [0.29; 0.36] | 3.1 | 0.109 | 0.411 [0.34; 0.49] |
| c_v10_noaug | 454 | 0.987 ± 0.000 [0.982; 0.994] | 15.4 | 0.0450 [0.0414; 0.0497] | 0.472 | 0.70 [0.69; 0.77] | 0.36 [0.31; 0.46] | 0.53 [0.48; 0.61] | 0.51 [0.48; 0.60] | 3.4 | 0.118 | 0.521 [0.44; 0.60] |
| g_v11_copypaste | 454 | 0.924 ± 0.002 [0.905; 0.951] | 10.7 | 0.0158 [0.0144; 0.0190] | 0.165 | 0.76 [0.73; 0.81] | 0.60 [0.53; 0.64] | 0.73 [0.66; 0.81] | 0.68 [0.65; 0.77] | 3.5 | 0.122 | 0.510 [0.43; 0.59] |
| j_v11_visible | 454 | 0.894 ± 0.003 [0.871; 0.923] | 10.8 | 0.0178 [0.0158; 0.0200] | 0.180 | 0.78 [0.75; 0.82] | 0.63 [0.57; 0.66] | 0.76 [0.69; 0.86] | 0.71 [0.67; 0.78] | 3.4 | 0.120 | 0.472 [0.40; 0.55] |
| m_pos | 454 | 0.912 ± 0.002 [0.897; 0.935] | 10.3 | 0.0135 [0.0117; 0.0165] | 0.142 | 0.75 [0.73; 0.80] | 0.65 [0.61; 0.70] | 0.71 [0.64; 0.85] | 0.67 [0.65; 0.75] | 3.6 | 0.123 | 0.486 [0.41; 0.58] |
| p_pos_tight | 454 | 0.902 ± 0.004 [0.885; 0.930] | 11.4 | 0.0215 [0.0186; 0.0258] | 0.219 | 0.78 [0.76; 0.82] | 0.63 [0.58; 0.68] | 0.75 [0.64; 0.82] | 0.67 [0.64; 0.72] | 3.3 | 0.112 | 0.548 [0.47; 0.62] |
| q_base | 454 | 0.901 ± 0.003 [0.877; 0.934] | 12.0 | 0.0255 [0.0215; 0.0304] | 0.249 | 0.80 [0.77; 0.84] | 0.55 [0.50; 0.61] | 0.84 [0.76; 0.93] | 0.67 [0.65; 0.72] | 3.2 | 0.106 | 0.479 [0.40; 0.54] |
| q_lab | 454 | 0.873 ± 0.001 [0.850; 0.909] | 11.8 | 0.0262 [0.0209; 0.0330] | 0.254 | 0.82 [0.81; 0.87] | 0.64 [0.60; 0.70] | 0.89 [0.82; 1.01] | 0.75 [0.70; 0.80] | 3.2 | 0.105 | 0.550 [0.50; 0.64] |
| q_lab_same | 454 | 0.864 ± 0.005 [0.836; 0.905] | 12.0 | 0.0267 [0.0225; 0.0342] | 0.259 | 0.81 [0.80; 0.86] | 0.62 [0.55; 0.66] | 0.89 [0.82; 1.01] | 0.71 [0.67; 0.78] | 3.2 | 0.104 | 0.547 [0.50; 0.63] |
| q_lab_local | 454 | 0.872 ± 0.003 [0.837; 0.911] | 12.1 | 0.0269 [0.0217; 0.0336] | 0.257 | 0.81 [0.80; 0.86] | 0.60 [0.55; 0.65] | 0.98 [0.92; 1.15] | 0.76 [0.72; 0.79] | 3.2 | 0.105 | 0.549 [0.48; 0.62] |
| r_matte | 454 | 0.903 ± 0.002 [0.882; 0.933] | 12.9 | 0.0318 [0.0270; 0.0413] | 0.301 | 0.82 [0.79; 0.85] | 0.61 [0.57; 0.64] | 0.99 [0.90; 1.14] | 0.70 [0.69; 0.76] | 3.1 | 0.102 | 0.508 [0.46; 0.57] |
| r_medium_harm | 454 | 0.852 ± 0.002 [0.817; 0.895] | 12.3 | 0.0289 [0.0220; 0.0338] | 0.273 | 0.83 [0.79; 0.87] | 0.65 [0.58; 0.70] | 0.97 [0.83; 1.10] | 0.71 [0.67; 0.75] | 3.2 | 0.103 | 0.502 [0.45; 0.57] |
| Synthetic Seabed (7 classes) | 401 | 1.000 ± 0.000 [1.000; 1.000] | 33.8 | 0.1313 [0.1286; 0.1358] | 1.323 | 0.08 [0.07; 0.18] | 0.00 [0.00; 0.00] | 0.02 [0.01; 0.04] | 0.03 [0.04; 0.06] | 2.6 | 0.080 | 0.329 [0.28; 0.44] |

### clip_b16: VAL reference (no shared TRAIN crops)

| Variant | FD | KID | CMMD | Precision | Recall | Density | Coverage | NN synth→real |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 7.0 | 0.0026 | 0.036 | 0.80 | 0.85 | 0.63 | 0.83 | 0.107 |
| ref_batch_v9 | 26.3 | 0.1101 | 1.112 | 0.59 | 0.15 | 0.37 | 0.22 | 0.149 |
| c_v10_noaug | 16.7 | 0.0577 | 0.583 | 0.70 | 0.46 | 0.42 | 0.42 | 0.132 |
| g_v11_copypaste | 9.9 | 0.0116 | 0.123 | 0.67 | 0.65 | 0.47 | 0.61 | 0.117 |
| j_v11_visible | 9.5 | 0.0103 | 0.109 | 0.65 | 0.68 | 0.47 | 0.64 | 0.112 |
| m_pos | 9.3 | 0.0087 | 0.094 | 0.68 | 0.67 | 0.47 | 0.59 | 0.118 |
| p_pos_tight | 9.9 | 0.0124 | 0.132 | 0.67 | 0.67 | 0.45 | 0.66 | 0.109 |
| q_base | 10.2 | 0.0149 | 0.151 | 0.69 | 0.62 | 0.50 | 0.63 | 0.104 |
| q_lab | 10.1 | 0.0149 | 0.150 | 0.69 | 0.62 | 0.52 | 0.66 | 0.103 |
| q_lab_same | 10.2 | 0.0152 | 0.153 | 0.68 | 0.64 | 0.53 | 0.66 | 0.102 |
| q_lab_local | 10.2 | 0.0155 | 0.154 | 0.69 | 0.61 | 0.60 | 0.64 | 0.102 |
| r_matte | 10.8 | 0.0189 | 0.183 | 0.70 | 0.62 | 0.55 | 0.63 | 0.101 |
| r_medium_harm | 10.3 | 0.0166 | 0.162 | 0.70 | 0.64 | 0.57 | 0.66 | 0.101 |
| Synthetic Seabed (7 classes) | 34.9 | 0.1408 | 1.403 | 0.10 | 0.00 | 0.03 | 0.04 | 0.176 |

### clip_b16: per-class AUC (out-of-fold scores of the global classifier)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.41 | 0.52 | 0.46 | 0.40 | 0.32 | 0.26 | 0.43 | 0.33 |
| ref_batch_v9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| c_v10_noaug | 0.99 | 0.97 | 1.00 | 1.00 | 0.98 | 0.98 | 0.99 | 1.00 |
| g_v11_copypaste | 0.90 | 0.94 | 0.97 | 0.95 | 0.94 | 0.87 | 0.90 | 0.96 |
| j_v11_visible | 0.84 | 0.85 | 0.94 | 0.93 | 0.90 | 0.86 | 0.91 | 0.97 |
| m_pos | 0.90 | 0.88 | 0.94 | 0.96 | 0.92 | 0.89 | 0.89 | 0.97 |
| p_pos_tight | 0.88 | 0.85 | 0.97 | 0.94 | 0.88 | 0.86 | 0.94 | 0.97 |
| q_base | 0.86 | 0.87 | 0.98 | 0.95 | 0.92 | 0.84 | 0.92 | 0.95 |
| q_lab | 0.85 | 0.91 | 0.96 | 0.90 | 0.86 | 0.84 | 0.85 | 0.91 |
| q_lab_same | 0.84 | 0.90 | 0.97 | 0.90 | 0.84 | 0.84 | 0.83 | 0.90 |
| q_lab_local | 0.86 | 0.83 | 0.92 | 0.97 | 0.88 | 0.81 | 0.84 | 0.94 |
| r_matte | 0.88 | 0.88 | 0.90 | 0.97 | 0.91 | 0.89 | 0.88 | 0.95 |
| r_medium_harm | 0.84 | 0.83 | 0.93 | 0.94 | 0.82 | 0.83 | 0.81 | 0.89 |
| Synthetic Seabed (7 classes) | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 1.00 | 1.00 |

### clip_b16: TSTR recall per class (train on pasted crops, evaluate on real VAL)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.74 | 0.51 | 0.68 | 0.83 | 0.72 | 0.57 | 0.78 | 0.75 |
| ref_batch_v9 | 0.03 | 0.69 | 0.61 | 0.60 | 0.60 | 0.26 | 0.17 | 0.33 |
| c_v10_noaug | 0.37 | 0.74 | 0.39 | 0.70 | 0.40 | 0.17 | 0.74 | 0.65 |
| g_v11_copypaste | 0.59 | 0.36 | 0.68 | 0.70 | 0.30 | 0.26 | 0.54 | 0.65 |
| j_v11_visible | 0.39 | 0.28 | 0.68 | 0.80 | 0.40 | 0.22 | 0.43 | 0.59 |
| m_pos | 0.29 | 0.54 | 0.61 | 0.70 | 0.40 | 0.17 | 0.52 | 0.67 |
| p_pos_tight | 0.40 | 0.51 | 0.61 | 0.83 | 0.64 | 0.17 | 0.54 | 0.68 |
| q_base | 0.50 | 0.30 | 0.57 | 0.67 | 0.43 | 0.17 | 0.54 | 0.65 |
| q_lab | 0.36 | 0.30 | 0.75 | 0.70 | 0.70 | 0.30 | 0.57 | 0.72 |
| q_lab_same | 0.39 | 0.28 | 0.75 | 0.62 | 0.75 | 0.26 | 0.57 | 0.75 |
| q_lab_local | 0.41 | 0.38 | 0.75 | 0.77 | 0.55 | 0.13 | 0.69 | 0.72 |
| r_matte | 0.56 | 0.23 | 0.71 | 0.52 | 0.43 | 0.13 | 0.74 | 0.74 |
| r_medium_harm | 0.27 | 0.38 | 0.68 | 0.60 | 0.70 | 0.17 | 0.54 | 0.68 |
| Synthetic Seabed (7 classes) | 0.39 | 0.13 | 0.29 | 0.45 | n/a | 0.26 | 0.30 | 0.49 |

### clip_b16: coverage per class (VAL reference)

| Variant | Bottle | Can | Fishing_Net | Glove | Mask | Metal_Debris | Plastic_Debris | Tire |
|---|---|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.71 | 0.56 | 0.96 | 0.92 | 0.87 | 0.65 | 0.94 | 0.86 |
| ref_batch_v9 | 0.16 | 0.28 | 0.43 | 0.17 | 0.19 | 0.48 | 0.44 | 0.19 |
| c_v10_noaug | 0.24 | 0.57 | 0.61 | 0.28 | 0.53 | 0.70 | 0.63 | 0.29 |
| g_v11_copypaste | 0.60 | 0.51 | 0.96 | 0.53 | 0.62 | 0.61 | 0.67 | 0.39 |
| j_v11_visible | 0.66 | 0.44 | 0.82 | 0.65 | 0.58 | 0.65 | 0.52 | 0.48 |
| m_pos | 0.59 | 0.59 | 0.93 | 0.50 | 0.43 | 0.48 | 0.52 | 0.43 |
| p_pos_tight | 0.63 | 0.57 | 0.93 | 0.53 | 0.75 | 0.65 | 0.56 | 0.45 |
| q_base | 0.60 | 0.61 | 0.89 | 0.50 | 0.75 | 0.61 | 0.50 | 0.54 |
| q_lab | 0.61 | 0.44 | 0.96 | 0.60 | 0.77 | 0.65 | 0.54 | 0.52 |
| q_lab_same | 0.61 | 0.44 | 0.93 | 0.62 | 0.77 | 0.61 | 0.52 | 0.51 |
| q_lab_local | 0.70 | 0.56 | 1.00 | 0.57 | 0.64 | 0.61 | 0.57 | 0.59 |
| r_matte | 0.63 | 0.56 | 0.96 | 0.57 | 0.77 | 0.39 | 0.56 | 0.59 |
| r_medium_harm | 0.69 | 0.59 | 0.93 | 0.63 | 0.79 | 0.43 | 0.54 | 0.58 |
| Synthetic Seabed (7 classes) | 0.11 | 0.05 | 0.18 | 0.03 | n/a | 0.30 | 0.20 | 0.01 |

## Same set of classes: 7 classes and common minimum quota (to include Synthetic Seabed)

Quotas: {'Bottle': 70, 'Can': 61, 'Fishing_Net': 28, 'Glove': 60, 'Mask': 0, 'Metal_Debris': 59, 'Plastic_Debris': 54, 'Tire': 69} (Mask = 0).

### resnet50

| Variant | Grouped AUC | KID | CMMD | Coverage@val | Density@val | TSTR bacc |
|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.393 ± 0.016 | 0.0034 | 0.047 | 0.74 | 0.61 | 0.686 |
| ref_batch_v9 | 0.991 ± 0.002 | 0.1226 | 0.945 | 0.29 | 0.68 | 0.362 |
| c_v10_noaug | 0.925 ± 0.008 | 0.0593 | 0.440 | 0.50 | 0.70 | 0.433 |
| g_v11_copypaste | 0.839 ± 0.012 | 0.0202 | 0.162 | 0.55 | 0.56 | 0.459 |
| j_v11_visible | 0.813 ± 0.004 | 0.0199 | 0.160 | 0.59 | 0.60 | 0.445 |
| m_pos | 0.799 ± 0.004 | 0.0136 | 0.117 | 0.56 | 0.62 | 0.405 |
| p_pos_tight | 0.840 ± 0.002 | 0.0261 | 0.217 | 0.62 | 0.57 | 0.434 |
| q_base | 0.816 ± 0.011 | 0.0256 | 0.220 | 0.57 | 0.56 | 0.490 |
| q_lab | 0.816 ± 0.007 | 0.0294 | 0.243 | 0.62 | 0.59 | 0.498 |
| q_lab_same | 0.815 ± 0.008 | 0.0308 | 0.252 | 0.64 | 0.60 | 0.450 |
| q_lab_local | 0.812 ± 0.010 | 0.0292 | 0.239 | 0.64 | 0.64 | 0.450 |
| r_matte | 0.818 ± 0.006 | 0.0327 | 0.263 | 0.62 | 0.57 | 0.379 |
| r_medium_harm | 0.793 ± 0.008 | 0.0292 | 0.237 | 0.63 | 0.59 | 0.480 |
| Synthetic Seabed (7 classes) | 1.000 ± 0.000 | 0.1084 | 0.783 | 0.18 | 0.27 | 0.258 |

### dinov2_b14

| Variant | Grouped AUC | KID | CMMD | Coverage@val | Density@val | TSTR bacc |
|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.448 ± 0.018 | 0.1417 | 0.066 | 0.83 | 0.67 | 0.765 |
| ref_batch_v9 | 0.997 ± 0.001 | 8.2066 | 1.845 | 0.25 | 0.33 | 0.500 |
| c_v10_noaug | 0.977 ± 0.001 | 6.0299 | 1.448 | 0.34 | 0.42 | 0.633 |
| g_v11_copypaste | 0.880 ± 0.003 | 1.2086 | 0.446 | 0.53 | 0.42 | 0.516 |
| j_v11_visible | 0.867 ± 0.001 | 1.0670 | 0.417 | 0.54 | 0.44 | 0.565 |
| m_pos | 0.868 ± 0.007 | 1.0838 | 0.416 | 0.47 | 0.38 | 0.480 |
| p_pos_tight | 0.914 ± 0.001 | 1.3905 | 0.583 | 0.41 | 0.37 | 0.648 |
| q_base | 0.891 ± 0.004 | 1.3240 | 0.562 | 0.48 | 0.43 | 0.611 |
| q_lab | 0.887 ± 0.003 | 1.3608 | 0.571 | 0.47 | 0.43 | 0.562 |
| q_lab_same | 0.891 ± 0.004 | 1.3227 | 0.556 | 0.45 | 0.43 | 0.614 |
| q_lab_local | 0.891 ± 0.003 | 1.3342 | 0.545 | 0.47 | 0.45 | 0.607 |
| r_matte | 0.913 ± 0.005 | 1.5525 | 0.657 | 0.45 | 0.44 | 0.495 |
| r_medium_harm | 0.873 ± 0.001 | 1.2805 | 0.536 | 0.45 | 0.43 | 0.559 |
| Synthetic Seabed (7 classes) | 1.000 ± 0.000 | 5.1692 | 1.669 | 0.09 | 0.10 | 0.475 |

### siglip2_b16

| Variant | Grouped AUC | KID | CMMD | Coverage@val | Density@val | TSTR bacc |
|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.380 ± 0.015 | 0.0012 | 0.030 | 0.84 | 0.66 | 0.754 |
| ref_batch_v9 | 0.999 ± 0.000 | 0.0606 | 0.900 | 0.25 | 0.29 | 0.536 |
| c_v10_noaug | 0.983 ± 0.003 | 0.0283 | 0.420 | 0.55 | 0.71 | 0.563 |
| g_v11_copypaste | 0.935 ± 0.002 | 0.0123 | 0.178 | 0.59 | 0.68 | 0.636 |
| j_v11_visible | 0.919 ± 0.006 | 0.0128 | 0.175 | 0.64 | 0.77 | 0.556 |
| m_pos | 0.933 ± 0.003 | 0.0092 | 0.140 | 0.61 | 0.69 | 0.589 |
| p_pos_tight | 0.915 ± 0.006 | 0.0164 | 0.227 | 0.65 | 0.66 | 0.637 |
| q_base | 0.911 ± 0.007 | 0.0189 | 0.245 | 0.60 | 0.73 | 0.620 |
| q_lab | 0.898 ± 0.001 | 0.0199 | 0.255 | 0.64 | 0.72 | 0.597 |
| q_lab_same | 0.894 ± 0.007 | 0.0208 | 0.264 | 0.60 | 0.70 | 0.601 |
| q_lab_local | 0.897 ± 0.004 | 0.0202 | 0.256 | 0.67 | 0.82 | 0.623 |
| r_matte | 0.916 ± 0.001 | 0.0239 | 0.287 | 0.68 | 0.85 | 0.501 |
| r_medium_harm | 0.891 ± 0.003 | 0.0197 | 0.251 | 0.62 | 0.71 | 0.584 |
| Synthetic Seabed (7 classes) | 1.000 ± 0.000 | 0.0678 | 0.987 | 0.05 | 0.03 | 0.368 |

### dinov3_b16

| Variant | Grouped AUC | KID | CMMD | Coverage@val | Density@val | TSTR bacc |
|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.421 ± 0.006 | 0.0029 | 0.065 | 0.76 | 0.63 | 0.761 |
| ref_batch_v9 | 0.999 ± 0.000 | 0.1935 | 1.731 | 0.22 | 0.25 | 0.528 |
| c_v10_noaug | 0.989 ± 0.001 | 0.1321 | 1.210 | 0.28 | 0.27 | 0.672 |
| g_v11_copypaste | 0.907 ± 0.005 | 0.0404 | 0.457 | 0.44 | 0.33 | 0.541 |
| j_v11_visible | 0.883 ± 0.007 | 0.0350 | 0.406 | 0.46 | 0.34 | 0.508 |
| m_pos | 0.911 ± 0.003 | 0.0333 | 0.388 | 0.38 | 0.29 | 0.547 |
| p_pos_tight | 0.922 ± 0.004 | 0.0452 | 0.515 | 0.39 | 0.32 | 0.602 |
| q_base | 0.924 ± 0.008 | 0.0463 | 0.527 | 0.38 | 0.29 | 0.651 |
| q_lab | 0.913 ± 0.002 | 0.0484 | 0.552 | 0.37 | 0.33 | 0.631 |
| q_lab_same | 0.922 ± 0.003 | 0.0491 | 0.558 | 0.36 | 0.32 | 0.611 |
| q_lab_local | 0.911 ± 0.004 | 0.0463 | 0.527 | 0.38 | 0.33 | 0.588 |
| r_matte | 0.921 ± 0.002 | 0.0527 | 0.596 | 0.39 | 0.34 | 0.549 |
| r_medium_harm | 0.923 ± 0.002 | 0.0469 | 0.537 | 0.38 | 0.33 | 0.585 |
| Synthetic Seabed (7 classes) | 1.000 ± 0.000 | 0.1978 | 2.126 | 0.07 | 0.04 | 0.463 |

### clip_b16

| Variant | Grouped AUC | KID | CMMD | Coverage@val | Density@val | TSTR bacc |
|---|---|---|---|---|---|---|
| real TRAIN vs VAL (floor) | 0.395 ± 0.016 | 0.0026 | 0.039 | 0.82 | 0.63 | 0.716 |
| ref_batch_v9 | 1.000 ± 0.000 | 0.0883 | 0.914 | 0.23 | 0.36 | 0.453 |
| c_v10_noaug | 0.988 ± 0.000 | 0.0446 | 0.470 | 0.42 | 0.43 | 0.541 |
| g_v11_copypaste | 0.919 ± 0.000 | 0.0159 | 0.169 | 0.60 | 0.47 | 0.554 |
| j_v11_visible | 0.888 ± 0.007 | 0.0181 | 0.187 | 0.65 | 0.49 | 0.489 |
| m_pos | 0.905 ± 0.003 | 0.0140 | 0.150 | 0.60 | 0.49 | 0.520 |
| p_pos_tight | 0.912 ± 0.002 | 0.0233 | 0.239 | 0.65 | 0.46 | 0.582 |
| q_base | 0.892 ± 0.008 | 0.0267 | 0.266 | 0.63 | 0.48 | 0.515 |
| q_lab | 0.878 ± 0.001 | 0.0285 | 0.279 | 0.64 | 0.51 | 0.571 |
| q_lab_same | 0.879 ± 0.005 | 0.0289 | 0.284 | 0.63 | 0.53 | 0.575 |
| q_lab_local | 0.870 ± 0.007 | 0.0289 | 0.278 | 0.65 | 0.60 | 0.579 |
| r_matte | 0.890 ± 0.005 | 0.0323 | 0.309 | 0.63 | 0.54 | 0.551 |
| r_medium_harm | 0.858 ± 0.005 | 0.0304 | 0.293 | 0.63 | 0.56 | 0.501 |
| Synthetic Seabed (7 classes) | 1.000 ± 0.000 | 0.1313 | 1.323 | 0.04 | 0.03 | 0.329 |

## Validation against the AP of the transfer check

**Warning:** 3 distinct batches and 5 arms: with so few points there is no reliable correlation (exact minimum Spearman p with n = 3: 0.333); the pairs are descriptive. With a single seed, the AP difference between batches may be noise.

Arms used: ['px_v9_real10', 'px_p_real10', 'px_q_real10', 'px_v9_real10_s2', 'px_p_real10_s2']. Warnings: none.

| Arm | Global AP | AP Bottle | AP Can | AP Fishing_Net | AP Glove | AP Mask | AP Metal_Debris | AP Plastic_Debris | AP Tire |
|---|---|---|---|---|---|---|---|---|---|
| px_v9_real10 | 35.00 | 29.92 | 11.07 | 36.40 | 64.00 | 56.09 | 4.58 | 61.13 | 16.78 |
| px_p_real10 | 36.61 | 31.87 | 12.66 | 34.01 | 67.95 | 58.96 | 5.52 | 64.36 | 17.53 |
| px_q_real10 | 36.09 | 31.34 | 13.59 | 32.74 | 67.34 | 57.60 | 4.55 | 63.57 | 18.00 |
| px_v9_real10_s2 | 34.04 | 30.53 | 10.60 | 34.22 | 62.57 | 53.65 | 3.05 | 61.44 | 16.26 |
| px_p_real10_s2 | 35.15 | 28.51 | 9.88 | 34.33 | 68.44 | 57.26 | 3.14 | 62.35 | 17.26 |

### resnet50: global agreement (global AP)

| Metric | Value per arm | Concordant pairs | Pairs with |dAP| ≥ 1 | Spearman |
|---|---|---|---|---|
| auc | px_p_real10=0.850, px_p_real10_s2=0.850, px_q_real10=0.836, px_v9_real10=0.996, px_v9_real10_s2=0.996 | 7/8 | 5/5 | 0.79 |
| fd | px_p_real10=65.274, px_p_real10_s2=65.274, px_q_real10=62.618, px_v9_real10=144.792, px_v9_real10_s2=144.792 | 7/8 | 5/5 | 0.79 |
| kid | px_p_real10=0.018, px_p_real10_s2=0.018, px_q_real10=0.018, px_v9_real10=0.129, px_v9_real10_s2=0.129 | 7/8 | 5/5 | 0.79 |
| cmmd | px_p_real10=0.141, px_p_real10_s2=0.141, px_q_real10=0.146, px_v9_real10=0.966, px_v9_real10_s2=0.966 | 7/8 | 5/5 | 0.79 |
| cmmd_unb | px_p_real10=0.133, px_p_real10_s2=0.133, px_q_real10=0.138, px_v9_real10=0.959, px_v9_real10_s2=0.959 | 7/8 | 5/5 | 0.79 |
| precision | px_p_real10=0.858, px_p_real10_s2=0.858, px_q_real10=0.869, px_v9_real10=0.810, px_v9_real10_s2=0.810 | 7/8 | 5/5 | 0.79 |
| recall | px_p_real10=0.698, px_p_real10_s2=0.698, px_q_real10=0.717, px_v9_real10=0.142, px_v9_real10_s2=0.142 | 7/8 | 5/5 | 0.79 |
| density | px_p_real10=0.709, px_p_real10_s2=0.709, px_q_real10=0.774, px_v9_real10=0.640, px_v9_real10_s2=0.640 | 7/8 | 5/5 | 0.79 |
| coverage | px_p_real10=0.700, px_p_real10_s2=0.700, px_q_real10=0.725, px_v9_real10=0.343, px_v9_real10_s2=0.343 | 7/8 | 5/5 | 0.79 |
| fd@val | px_p_real10=71.383, px_p_real10_s2=71.383, px_q_real10=66.667, px_v9_real10=164.399, px_v9_real10_s2=164.399 | 7/8 | 5/5 | 0.79 |
| kid@val | px_p_real10=0.016, px_p_real10_s2=0.016, px_q_real10=0.015, px_v9_real10=0.151, px_v9_real10_s2=0.151 | 7/8 | 5/5 | 0.79 |
| cmmd@val | px_p_real10=0.119, px_p_real10_s2=0.119, px_q_real10=0.113, px_v9_real10=1.140, px_v9_real10_s2=1.140 | 7/8 | 5/5 | 0.79 |
| precision@val | px_p_real10=0.735, px_p_real10_s2=0.735, px_q_real10=0.779, px_v9_real10=0.709, px_v9_real10_s2=0.709 | 7/8 | 5/5 | 0.79 |
| recall@val | px_p_real10=0.728, px_p_real10_s2=0.728, px_q_real10=0.719, px_v9_real10=0.163, px_v9_real10_s2=0.163 | 7/8 | 5/5 | 0.79 |
| density@val | px_p_real10=0.442, px_p_real10_s2=0.442, px_q_real10=0.494, px_v9_real10=0.468, px_v9_real10_s2=0.468 | 3/8 | 2/5 | -0.26 |
| coverage@val | px_p_real10=0.490, px_p_real10_s2=0.490, px_q_real10=0.525, px_v9_real10=0.187, px_v9_real10_s2=0.187 | 7/8 | 5/5 | 0.79 |
| nn_real | px_p_real10=0.195, px_p_real10_s2=0.195, px_q_real10=0.191, px_v9_real10=0.195, px_v9_real10_s2=0.195 | 5/8 | 3/5 | 0.26 |
| vendi | px_p_real10=7.658, px_p_real10_s2=7.658, px_q_real10=7.778, px_v9_real10=5.298, px_v9_real10_s2=5.298 | 7/8 | 5/5 | 0.79 |
| nn_dist | px_p_real10=0.191, px_p_real10_s2=0.191, px_q_real10=0.190, px_v9_real10=0.157, px_v9_real10_s2=0.157 | 7/8 | 5/5 | 0.79 |
| tstr_bacc | px_p_real10=0.482, px_p_real10_s2=0.482, px_q_real10=0.441, px_v9_real10=0.348, px_v9_real10_s2=0.348 | 7/8 | 5/5 | 0.79 |

### resnet50: per-class agreement (8 classes × batches; the pairs of one class are not independent)

| Metric | Concordant pairs (all classes) | Pairs with |dAP class| ≥ 1 | Mean Spearman per class |
|---|---|---|---|
| auc | 47/64 | 34/46 | 0.49 |
| kid | 45/64 | 32/46 | 0.43 |
| cmmd | 45/64 | 33/46 | 0.42 |
| precision | 43/64 | 33/46 | 0.35 |
| recall | 47/64 | 35/46 | 0.51 |
| density | 37/64 | 27/46 | 0.18 |
| coverage | 43/64 | 32/46 | 0.35 |
| kid@val | 45/64 | 32/46 | 0.42 |
| cmmd@val | 47/64 | 34/46 | 0.48 |
| precision@val | 37/64 | 28/46 | 0.15 |
| recall@val | 47/64 | 35/46 | 0.49 |
| density@val | 35/64 | 26/46 | 0.13 |
| coverage@val | 42/62 | 31/45 | 0.36 |
| vendi | 49/64 | 37/46 | 0.53 |
| tstr_recall | 32/60 | 21/43 | 0.05 |

### dinov2_b14: global agreement (global AP)

| Metric | Value per arm | Concordant pairs | Pairs with |dAP| ≥ 1 | Spearman |
|---|---|---|---|---|
| auc | px_p_real10=0.928, px_p_real10_s2=0.928, px_q_real10=0.913, px_v9_real10=0.998, px_v9_real10_s2=0.998 | 7/8 | 5/5 | 0.79 |
| fd | px_p_real10=613.375, px_p_real10_s2=613.375, px_q_real10=597.630, px_v9_real10=1201.313, px_v9_real10_s2=1201.313 | 7/8 | 5/5 | 0.79 |
| kid | px_p_real10=1.340, px_p_real10_s2=1.340, px_q_real10=1.209, px_v9_real10=8.339, px_v9_real10_s2=8.339 | 7/8 | 5/5 | 0.79 |
| cmmd | px_p_real10=0.544, px_p_real10_s2=0.544, px_q_real10=0.505, px_v9_real10=1.869, px_v9_real10_s2=1.869 | 7/8 | 5/5 | 0.79 |
| cmmd_unb | px_p_real10=0.529, px_p_real10_s2=0.529, px_q_real10=0.489, px_v9_real10=1.856, px_v9_real10_s2=1.856 | 7/8 | 5/5 | 0.79 |
| precision | px_p_real10=0.689, px_p_real10_s2=0.689, px_q_real10=0.730, px_v9_real10=0.698, px_v9_real10_s2=0.698 | 3/8 | 2/5 | -0.26 |
| recall | px_p_real10=0.742, px_p_real10_s2=0.742, px_q_real10=0.752, px_v9_real10=0.174, px_v9_real10_s2=0.174 | 7/8 | 5/5 | 0.79 |
| density | px_p_real10=0.455, px_p_real10_s2=0.455, px_q_real10=0.519, px_v9_real10=0.506, px_v9_real10_s2=0.506 | 3/8 | 2/5 | -0.26 |
| coverage | px_p_real10=0.468, px_p_real10_s2=0.468, px_q_real10=0.546, px_v9_real10=0.269, px_v9_real10_s2=0.269 | 7/8 | 5/5 | 0.79 |
| fd@val | px_p_real10=665.562, px_p_real10_s2=665.562, px_q_real10=649.267, px_v9_real10=1287.358, px_v9_real10_s2=1287.358 | 7/8 | 5/5 | 0.79 |
| kid@val | px_p_real10=1.357, px_p_real10_s2=1.357, px_q_real10=1.222, px_v9_real10=8.737, px_v9_real10_s2=8.737 | 7/8 | 5/5 | 0.79 |
| cmmd@val | px_p_real10=0.517, px_p_real10_s2=0.517, px_q_real10=0.475, px_v9_real10=2.011, px_v9_real10_s2=2.011 | 7/8 | 5/5 | 0.79 |
| precision@val | px_p_real10=0.597, px_p_real10_s2=0.597, px_q_real10=0.647, px_v9_real10=0.651, px_v9_real10_s2=0.651 | 1/8 | 0/5 | -0.79 |
| recall@val | px_p_real10=0.717, px_p_real10_s2=0.717, px_q_real10=0.715, px_v9_real10=0.178, px_v9_real10_s2=0.178 | 7/8 | 5/5 | 0.79 |
| density@val | px_p_real10=0.355, px_p_real10_s2=0.355, px_q_real10=0.403, px_v9_real10=0.353, px_v9_real10_s2=0.353 | 7/8 | 5/5 | 0.79 |
| coverage@val | px_p_real10=0.365, px_p_real10_s2=0.365, px_q_real10=0.433, px_v9_real10=0.201, px_v9_real10_s2=0.201 | 7/8 | 5/5 | 0.79 |
| nn_real | px_p_real10=0.450, px_p_real10_s2=0.450, px_q_real10=0.442, px_v9_real10=0.389, px_v9_real10_s2=0.389 | 7/8 | 5/5 | 0.79 |
| vendi | px_p_real10=34.961, px_p_real10_s2=34.961, px_q_real10=35.757, px_v9_real10=15.977, px_v9_real10_s2=15.977 | 7/8 | 5/5 | 0.79 |
| nn_dist | px_p_real10=0.440, px_p_real10_s2=0.440, px_q_real10=0.441, px_v9_real10=0.288, px_v9_real10_s2=0.288 | 7/8 | 5/5 | 0.79 |
| tstr_bacc | px_p_real10=0.611, px_p_real10_s2=0.611, px_q_real10=0.611, px_v9_real10=0.488, px_v9_real10_s2=0.488 | 7/8 | 5/5 | 0.79 |

### dinov2_b14: per-class agreement (8 classes × batches; the pairs of one class are not independent)

| Metric | Concordant pairs (all classes) | Pairs with |dAP class| ≥ 1 | Mean Spearman per class |
|---|---|---|---|
| auc | 45/64 | 34/46 | 0.42 |
| kid | 43/64 | 32/46 | 0.35 |
| cmmd | 43/64 | 31/46 | 0.36 |
| precision | 29/60 | 20/42 | -0.02 |
| recall | 47/64 | 35/46 | 0.48 |
| density | 25/64 | 18/46 | -0.24 |
| coverage | 41/64 | 32/46 | 0.28 |
| kid@val | 47/64 | 34/46 | 0.45 |
| cmmd@val | 45/64 | 33/46 | 0.39 |
| precision@val | 31/64 | 23/46 | 0.02 |
| recall@val | 47/64 | 35/46 | 0.48 |
| density@val | 33/64 | 23/46 | 0.05 |
| coverage@val | 36/60 | 26/44 | 0.17 |
| vendi | 47/64 | 34/46 | 0.45 |
| tstr_recall | 39/64 | 28/46 | 0.23 |

### siglip2_b16: global agreement (global AP)

| Metric | Value per arm | Concordant pairs | Pairs with |dAP| ≥ 1 | Spearman |
|---|---|---|---|---|
| auc | px_p_real10=0.928, px_p_real10_s2=0.928, px_q_real10=0.915, px_v9_real10=0.999, px_v9_real10_s2=0.999 | 7/8 | 5/5 | 0.79 |
| fd | px_p_real10=11.219, px_p_real10_s2=11.219, px_q_real10=11.294, px_v9_real10=25.015, px_v9_real10_s2=25.015 | 7/8 | 5/5 | 0.79 |
| kid | px_p_real10=0.015, px_p_real10_s2=0.015, px_q_real10=0.016, px_v9_real10=0.058, px_v9_real10_s2=0.058 | 7/8 | 5/5 | 0.79 |
| cmmd | px_p_real10=0.205, px_p_real10_s2=0.205, px_q_real10=0.215, px_v9_real10=0.856, px_v9_real10_s2=0.856 | 7/8 | 5/5 | 0.79 |
| cmmd_unb | px_p_real10=0.200, px_p_real10_s2=0.200, px_q_real10=0.210, px_v9_real10=0.851, px_v9_real10_s2=0.851 | 7/8 | 5/5 | 0.79 |
| precision | px_p_real10=0.695, px_p_real10_s2=0.695, px_q_real10=0.779, px_v9_real10=0.415, px_v9_real10_s2=0.415 | 7/8 | 5/5 | 0.79 |
| recall | px_p_real10=0.570, px_p_real10_s2=0.570, px_q_real10=0.610, px_v9_real10=0.181, px_v9_real10_s2=0.181 | 7/8 | 5/5 | 0.79 |
| density | px_p_real10=0.660, px_p_real10_s2=0.660, px_q_real10=0.880, px_v9_real10=0.205, px_v9_real10_s2=0.205 | 7/8 | 5/5 | 0.79 |
| coverage | px_p_real10=0.584, px_p_real10_s2=0.584, px_q_real10=0.657, px_v9_real10=0.258, px_v9_real10_s2=0.258 | 7/8 | 5/5 | 0.79 |
| fd@val | px_p_real10=10.987, px_p_real10_s2=10.987, px_q_real10=10.838, px_v9_real10=28.832, px_v9_real10_s2=28.832 | 7/8 | 5/5 | 0.79 |
| kid@val | px_p_real10=0.011, px_p_real10_s2=0.011, px_q_real10=0.012, px_v9_real10=0.069, px_v9_real10_s2=0.069 | 7/8 | 5/5 | 0.79 |
| cmmd@val | px_p_real10=0.163, px_p_real10_s2=0.163, px_q_real10=0.169, px_v9_real10=1.010, px_v9_real10_s2=1.010 | 7/8 | 5/5 | 0.79 |
| precision@val | px_p_real10=0.621, px_p_real10_s2=0.621, px_q_real10=0.712, px_v9_real10=0.426, px_v9_real10_s2=0.426 | 7/8 | 5/5 | 0.79 |
| recall@val | px_p_real10=0.616, px_p_real10_s2=0.616, px_q_real10=0.609, px_v9_real10=0.152, px_v9_real10_s2=0.152 | 7/8 | 5/5 | 0.79 |
| density@val | px_p_real10=0.468, px_p_real10_s2=0.468, px_q_real10=0.634, px_v9_real10=0.217, px_v9_real10_s2=0.217 | 7/8 | 5/5 | 0.79 |
| coverage@val | px_p_real10=0.516, px_p_real10_s2=0.516, px_q_real10=0.535, px_v9_real10=0.189, px_v9_real10_s2=0.189 | 7/8 | 5/5 | 0.79 |
| nn_real | px_p_real10=0.115, px_p_real10_s2=0.115, px_q_real10=0.105, px_v9_real10=0.147, px_v9_real10_s2=0.147 | 1/8 | 0/5 | -0.79 |
| vendi | px_p_real10=3.886, px_p_real10_s2=3.886, px_q_real10=3.674, px_v9_real10=3.539, px_v9_real10_s2=3.539 | 7/8 | 5/5 | 0.79 |
| nn_dist | px_p_real10=0.110, px_p_real10_s2=0.110, px_q_real10=0.102, px_v9_real10=0.106, px_v9_real10_s2=0.106 | 5/8 | 3/5 | 0.26 |
| tstr_bacc | px_p_real10=0.559, px_p_real10_s2=0.559, px_q_real10=0.611, px_v9_real10=0.570, px_v9_real10_s2=0.570 | 3/8 | 2/5 | -0.26 |

### siglip2_b16: per-class agreement (8 classes × batches; the pairs of one class are not independent)

| Metric | Concordant pairs (all classes) | Pairs with |dAP class| ≥ 1 | Mean Spearman per class |
|---|---|---|---|
| auc | 43/64 | 35/46 | 0.40 |
| kid | 45/64 | 33/46 | 0.40 |
| cmmd | 45/64 | 33/46 | 0.39 |
| precision | 47/64 | 36/46 | 0.49 |
| recall | 45/64 | 32/46 | 0.42 |
| density | 47/64 | 36/46 | 0.49 |
| coverage | 45/64 | 34/46 | 0.44 |
| kid@val | 39/64 | 29/46 | 0.19 |
| cmmd@val | 41/64 | 31/46 | 0.26 |
| precision@val | 40/62 | 31/45 | 0.31 |
| recall@val | 42/60 | 30/44 | 0.41 |
| density@val | 45/64 | 33/46 | 0.44 |
| coverage@val | 44/62 | 33/45 | 0.42 |
| vendi | 41/64 | 28/46 | 0.28 |
| tstr_recall | 28/62 | 19/45 | -0.08 |

### dinov3_b16: global agreement (global AP)

| Metric | Value per arm | Concordant pairs | Pairs with |dAP| ≥ 1 | Spearman |
|---|---|---|---|---|
| auc | px_p_real10=0.941, px_p_real10_s2=0.941, px_q_real10=0.929, px_v9_real10=1.000, px_v9_real10_s2=1.000 | 7/8 | 5/5 | 0.79 |
| fd | px_p_real10=54.904, px_p_real10_s2=54.904, px_q_real10=55.085, px_v9_real10=108.878, px_v9_real10_s2=108.878 | 7/8 | 5/5 | 0.79 |
| kid | px_p_real10=0.040, px_p_real10_s2=0.040, px_q_real10=0.040, px_v9_real10=0.193, px_v9_real10_s2=0.193 | 7/8 | 5/5 | 0.79 |
| cmmd | px_p_real10=0.444, px_p_real10_s2=0.444, px_q_real10=0.444, px_v9_real10=1.707, px_v9_real10_s2=1.707 | 7/8 | 5/5 | 0.79 |
| cmmd_unb | px_p_real10=0.428, px_p_real10_s2=0.428, px_q_real10=0.428, px_v9_real10=1.693, px_v9_real10_s2=1.693 | 7/8 | 5/5 | 0.79 |
| precision | px_p_real10=0.685, px_p_real10_s2=0.685, px_q_real10=0.695, px_v9_real10=0.596, px_v9_real10_s2=0.596 | 7/8 | 5/5 | 0.79 |
| recall | px_p_real10=0.784, px_p_real10_s2=0.784, px_q_real10=0.772, px_v9_real10=0.142, px_v9_real10_s2=0.142 | 7/8 | 5/5 | 0.79 |
| density | px_p_real10=0.360, px_p_real10_s2=0.360, px_q_real10=0.390, px_v9_real10=0.266, px_v9_real10_s2=0.266 | 7/8 | 5/5 | 0.79 |
| coverage | px_p_real10=0.494, px_p_real10_s2=0.494, px_q_real10=0.503, px_v9_real10=0.226, px_v9_real10_s2=0.226 | 7/8 | 5/5 | 0.79 |
| fd@val | px_p_real10=62.832, px_p_real10_s2=62.832, px_q_real10=62.417, px_v9_real10=118.704, px_v9_real10_s2=118.704 | 7/8 | 5/5 | 0.79 |
| kid@val | px_p_real10=0.042, px_p_real10_s2=0.042, px_q_real10=0.042, px_v9_real10=0.203, px_v9_real10_s2=0.203 | 7/8 | 5/5 | 0.79 |
| cmmd@val | px_p_real10=0.465, px_p_real10_s2=0.465, px_q_real10=0.458, px_v9_real10=1.811, px_v9_real10_s2=1.811 | 7/8 | 5/5 | 0.79 |
| precision@val | px_p_real10=0.505, px_p_real10_s2=0.505, px_q_real10=0.540, px_v9_real10=0.358, px_v9_real10_s2=0.358 | 7/8 | 5/5 | 0.79 |
| recall@val | px_p_real10=0.721, px_p_real10_s2=0.721, px_q_real10=0.762, px_v9_real10=0.124, px_v9_real10_s2=0.124 | 7/8 | 5/5 | 0.79 |
| density@val | px_p_real10=0.233, px_p_real10_s2=0.233, px_q_real10=0.269, px_v9_real10=0.128, px_v9_real10_s2=0.128 | 7/8 | 5/5 | 0.79 |
| coverage@val | px_p_real10=0.309, px_p_real10_s2=0.309, px_q_real10=0.368, px_v9_real10=0.159, px_v9_real10_s2=0.159 | 7/8 | 5/5 | 0.79 |
| nn_real | px_p_real10=0.482, px_p_real10_s2=0.482, px_q_real10=0.475, px_v9_real10=0.463, px_v9_real10_s2=0.463 | 7/8 | 5/5 | 0.79 |
| vendi | px_p_real10=39.133, px_p_real10_s2=39.133, px_q_real10=39.155, px_v9_real10=21.007, px_v9_real10_s2=21.007 | 7/8 | 5/5 | 0.79 |
| nn_dist | px_p_real10=0.468, px_p_real10_s2=0.468, px_q_real10=0.464, px_v9_real10=0.346, px_v9_real10_s2=0.346 | 7/8 | 5/5 | 0.79 |
| tstr_bacc | px_p_real10=0.647, px_p_real10_s2=0.647, px_q_real10=0.592, px_v9_real10=0.495, px_v9_real10_s2=0.495 | 7/8 | 5/5 | 0.79 |

### dinov3_b16: per-class agreement (8 classes × batches; the pairs of one class are not independent)

| Metric | Concordant pairs (all classes) | Pairs with |dAP class| ≥ 1 | Mean Spearman per class |
|---|---|---|---|
| auc | 43/64 | 32/46 | 0.35 |
| kid | 45/64 | 32/46 | 0.40 |
| cmmd | 45/64 | 32/46 | 0.40 |
| precision | 43/64 | 34/46 | 0.40 |
| recall | 44/60 | 32/43 | 0.43 |
| density | 45/64 | 34/46 | 0.44 |
| coverage | 40/58 | 28/43 | 0.35 |
| kid@val | 45/64 | 32/46 | 0.40 |
| cmmd@val | 43/64 | 31/46 | 0.34 |
| precision@val | 35/64 | 24/46 | 0.13 |
| recall@val | 49/64 | 35/46 | 0.51 |
| density@val | 35/62 | 26/45 | 0.14 |
| coverage@val | 45/64 | 33/46 | 0.42 |
| vendi | 45/64 | 32/46 | 0.39 |
| tstr_recall | 41/64 | 30/46 | 0.24 |

### clip_b16: global agreement (global AP)

| Metric | Value per arm | Concordant pairs | Pairs with |dAP| ≥ 1 | Spearman |
|---|---|---|---|---|
| auc | px_p_real10=0.931, px_p_real10_s2=0.931, px_q_real10=0.909, px_v9_real10=1.000, px_v9_real10_s2=1.000 | 7/8 | 5/5 | 0.79 |
| fd | px_p_real10=8.437, px_p_real10_s2=8.437, px_q_real10=8.853, px_v9_real10=21.065, px_v9_real10_s2=21.065 | 7/8 | 5/5 | 0.79 |
| kid | px_p_real10=0.021, px_p_real10_s2=0.021, px_q_real10=0.025, px_v9_real10=0.088, px_v9_real10_s2=0.088 | 7/8 | 5/5 | 0.79 |
| cmmd | px_p_real10=0.205, px_p_real10_s2=0.205, px_q_real10=0.239, px_v9_real10=0.908, px_v9_real10_s2=0.908 | 7/8 | 5/5 | 0.79 |
| cmmd_unb | px_p_real10=0.199, px_p_real10_s2=0.199, px_q_real10=0.234, px_v9_real10=0.903, px_v9_real10_s2=0.903 | 7/8 | 5/5 | 0.79 |
| precision | px_p_real10=0.691, px_p_real10_s2=0.691, px_q_real10=0.782, px_v9_real10=0.429, px_v9_real10_s2=0.429 | 7/8 | 5/5 | 0.79 |
| recall | px_p_real10=0.624, px_p_real10_s2=0.624, px_q_real10=0.619, px_v9_real10=0.083, px_v9_real10_s2=0.083 | 7/8 | 5/5 | 0.79 |
| density | px_p_real10=0.642, px_p_real10_s2=0.642, px_q_real10=0.750, px_v9_real10=0.198, px_v9_real10_s2=0.198 | 7/8 | 5/5 | 0.79 |
| coverage | px_p_real10=0.644, px_p_real10_s2=0.644, px_q_real10=0.688, px_v9_real10=0.212, px_v9_real10_s2=0.212 | 7/8 | 5/5 | 0.79 |
| fd@val | px_p_real10=7.753, px_p_real10_s2=7.753, px_q_real10=7.922, px_v9_real10=24.298, px_v9_real10_s2=24.298 | 7/8 | 5/5 | 0.79 |
| kid@val | px_p_real10=0.013, px_p_real10_s2=0.013, px_q_real10=0.017, px_v9_real10=0.108, px_v9_real10_s2=0.108 | 7/8 | 5/5 | 0.79 |
| cmmd@val | px_p_real10=0.135, px_p_real10_s2=0.135, px_q_real10=0.162, px_v9_real10=1.094, px_v9_real10_s2=1.094 | 7/8 | 5/5 | 0.79 |
| precision@val | px_p_real10=0.579, px_p_real10_s2=0.579, px_q_real10=0.655, px_v9_real10=0.449, px_v9_real10_s2=0.449 | 7/8 | 5/5 | 0.79 |
| recall@val | px_p_real10=0.653, px_p_real10_s2=0.653, px_q_real10=0.655, px_v9_real10=0.091, px_v9_real10_s2=0.091 | 7/8 | 5/5 | 0.79 |
| density@val | px_p_real10=0.391, px_p_real10_s2=0.391, px_q_real10=0.457, px_v9_real10=0.206, px_v9_real10_s2=0.206 | 7/8 | 5/5 | 0.79 |
| coverage@val | px_p_real10=0.556, px_p_real10_s2=0.556, px_q_real10=0.584, px_v9_real10=0.154, px_v9_real10_s2=0.154 | 7/8 | 5/5 | 0.79 |
| nn_real | px_p_real10=0.109, px_p_real10_s2=0.109, px_q_real10=0.101, px_v9_real10=0.143, px_v9_real10_s2=0.143 | 1/8 | 0/5 | -0.79 |
| vendi | px_p_real10=3.846, px_p_real10_s2=3.846, px_q_real10=3.629, px_v9_real10=3.348, px_v9_real10_s2=3.348 | 7/8 | 5/5 | 0.79 |
| nn_dist | px_p_real10=0.108, px_p_real10_s2=0.108, px_q_real10=0.102, px_v9_real10=0.099, px_v9_real10_s2=0.099 | 7/8 | 5/5 | 0.79 |
| tstr_bacc | px_p_real10=0.569, px_p_real10_s2=0.569, px_q_real10=0.513, px_v9_real10=0.382, px_v9_real10_s2=0.382 | 7/8 | 5/5 | 0.79 |

### clip_b16: per-class agreement (8 classes × batches; the pairs of one class are not independent)

| Metric | Concordant pairs (all classes) | Pairs with |dAP class| ≥ 1 | Mean Spearman per class |
|---|---|---|---|
| auc | 45/64 | 34/46 | 0.42 |
| kid | 39/64 | 25/46 | 0.19 |
| cmmd | 43/64 | 28/46 | 0.32 |
| precision | 45/64 | 34/46 | 0.44 |
| recall | 45/64 | 33/46 | 0.42 |
| density | 45/64 | 33/46 | 0.44 |
| coverage | 45/64 | 34/46 | 0.44 |
| kid@val | 39/64 | 27/46 | 0.20 |
| cmmd@val | 41/64 | 29/46 | 0.27 |
| precision@val | 35/58 | 27/41 | 0.18 |
| recall@val | 45/62 | 35/46 | 0.44 |
| density@val | 43/64 | 33/46 | 0.36 |
| coverage@val | 45/64 | 32/46 | 0.42 |
| vendi | 41/64 | 29/46 | 0.30 |
| tstr_recall | 31/60 | 20/43 | 0.02 |

