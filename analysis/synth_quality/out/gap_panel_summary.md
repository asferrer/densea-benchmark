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
