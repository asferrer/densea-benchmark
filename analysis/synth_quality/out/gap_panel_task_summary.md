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
