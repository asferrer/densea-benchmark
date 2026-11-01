# Pilot variants of the copy-paste batch (v11): small pilot, CPU

Images per variant: 300. Objects per class (common quota): {'Bottle': 70, 'Can': 61, 'Fishing_Net': 28, 'Glove': 60, 'Mask': 53, 'Metal_Debris': 59, 'Plastic_Debris': 54, 'Tire': 69}. Real: same TRAIN subset for all variants. AUC: mean ± standard deviation over 3 CV seeds (5 folds). **Main column: AUC with GROUPED CV** (StratifiedGroupKFold; groups in the "Groups" section); the AUC with stratified CV without grouping is kept next to it as a reference. Frechet on ResNet-50 (2048 d) with small n: biased upwards, only comparable between variants of this same report.

| Variant | AUC img grouped | AUC img ungrouped | Frechet img | AUC obj grouped | AUC obj ungrouped | Frechet obj | Mean Vendi | Mean NN |
|---|---|---|---|---|---|---|---|---|
| ref_batch_v9 | 0.997 ± 0.001 | 0.998 ± 0.000 | 206.1 | 0.991 ± 0.001 | 0.994 ± 0.000 | 170.1 | 4.77 | 0.1724 |
| c_v10_noaug | 0.960 ± 0.003 | 0.964 ± 0.004 | 174.8 | 0.937 ± 0.002 | 0.952 ± 0.003 | 121.1 | 5.37 | 0.1892 |
| g_v11_copypaste | 0.634 ± 0.010 | 0.644 ± 0.005 | 89.7 | 0.839 ± 0.003 | 0.873 ± 0.004 | 94.1 | 6.47 | 0.2076 |
| j_v11_visible | 0.613 ± 0.017 | 0.620 ± 0.007 | 90.0 | 0.811 ± 0.002 | 0.853 ± 0.008 | 92.3 | 6.51 | 0.2089 |
| m_pos | 0.633 ± 0.002 | 0.614 ± 0.003 | 92.6 | 0.809 ± 0.006 | 0.840 ± 0.003 | 88.4 | 6.48 | 0.2124 |
| p_pos_tight | 0.634 ± 0.007 | 0.611 ± 0.012 | 91.5 | 0.836 ± 0.003 | 0.864 ± 0.005 | 95.3 | 6.19 | 0.2005 |
| q_base | 0.633 ± 0.016 | 0.612 ± 0.013 | 90.9 | 0.821 ± 0.002 | 0.861 ± 0.003 | 95.0 | 6.37 | 0.1991 |
| q_lab | 0.609 ± 0.003 | 0.595 ± 0.006 | 91.3 | 0.810 ± 0.005 | 0.844 ± 0.006 | 94.2 | 6.29 | 0.1940 |
| q_lab_same | 0.611 ± 0.004 | 0.593 ± 0.003 | 91.4 | 0.796 ± 0.002 | 0.837 ± 0.002 | 96.3 | 6.35 | 0.1962 |
| q_lab_local | 0.603 ± 0.008 | 0.591 ± 0.005 | 90.2 | 0.813 ± 0.004 | 0.850 ± 0.000 | 94.7 | 6.34 | 0.1987 |
| r_matte | 0.618 ± 0.007 | 0.586 ± 0.008 | 91.6 | 0.837 ± 0.012 | 0.866 ± 0.005 | 100.9 | 6.51 | 0.1990 |
| r_medium_harm | 0.603 ± 0.009 | 0.610 ± 0.014 | 91.3 | 0.794 ± 0.008 | 0.829 ± 0.007 | 94.7 | 6.34 | 0.1969 |

Control real TRAIN vs VAL (grouped | ungrouped): img 0.477 ± 0.007 | 0.656 ± 0.008; obj 0.405 ± 0.010 | 0.627 ± 0.012. Frechet real TRAIN-VAL: img 106.0, obj 79.8.
Diversity of real TRAIN objects (same n per class): mean Vendi 6.44, mean NN 0.1908.

## Groups of the grouped CV

- Real images (TRAIN and VAL): partition group of the image; the objects of an image share a group.
- Synthetic with a real TRAIN background: partition group of the real background image (COCO field `background`), so synthetic and real items of the same group are never split across folds.
- Synthetic without a real background: one group per reused background or, without identifier, one per synthetic image.

| Variant | Criterion | Traceable backgrounds | Images | Groups (with real) |
|---|---|---|---|---|
| ref_batch_v9 | one group per synthetic image (no background identifier) | 0 | 738 | 696 |
| c_v10_noaug | one group per background (not a real TRAIN image) | 0 | 300 | 513 |
| g_v11_copypaste | partition group of the real background image | 300 | 300 | 383 |
| j_v11_visible | partition group of the real background image | 300 | 300 | 390 |
| m_pos | partition group of the real background image | 300 | 300 | 379 |
| p_pos_tight | partition group of the real background image | 300 | 300 | 382 |
| q_base | partition group of the real background image | 300 | 300 | 384 |
| q_lab | partition group of the real background image | 300 | 300 | 382 |
| q_lab_same | partition group of the real background image | 300 | 300 | 385 |
| q_lab_local | partition group of the real background image | 300 | 300 | 379 |
| r_matte | partition group of the real background image | 300 | 300 | 374 |
| r_medium_harm | partition group of the real background image | 300 | 300 | 385 |

## Diversity per class (Vendi / NN)

| Class | n | real | ref_batch_v9 | c_v10_noaug | g_v11_copypaste | j_v11_visible | m_pos | p_pos_tight | q_base | q_lab | q_lab_same | q_lab_local | r_matte | r_medium_harm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bottle | 70 | 7.5 / 0.203 | 5.3 / 0.178 | 5.7 / 0.192 | 6.8 / 0.205 | 7.1 / 0.201 | 6.9 / 0.214 | 6.9 / 0.203 | 7.2 / 0.200 | 7.0 / 0.205 | 7.1 / 0.205 | 7.4 / 0.203 | 7.4 / 0.203 | 7.3 / 0.210 |
| Can | 61 | 7.1 / 0.209 | 5.1 / 0.183 | 5.9 / 0.196 | 7.2 / 0.216 | 7.2 / 0.206 | 6.9 / 0.207 | 5.7 / 0.166 | 5.9 / 0.171 | 5.9 / 0.171 | 5.7 / 0.164 | 5.8 / 0.171 | 6.3 / 0.177 | 6.0 / 0.180 |
| Fishing_Net | 28 | 5.3 / 0.229 | 3.6 / 0.164 | 3.9 / 0.180 | 4.4 / 0.200 | 4.4 / 0.206 | 4.3 / 0.209 | 4.4 / 0.203 | 4.3 / 0.188 | 4.4 / 0.186 | 4.8 / 0.208 | 4.7 / 0.212 | 4.7 / 0.209 | 4.5 / 0.195 |
| Glove | 60 | 5.8 / 0.161 | 4.6 / 0.160 | 5.2 / 0.178 | 6.4 / 0.209 | 6.7 / 0.209 | 6.3 / 0.202 | 6.4 / 0.202 | 6.5 / 0.206 | 6.5 / 0.192 | 6.4 / 0.190 | 6.4 / 0.195 | 6.9 / 0.201 | 6.5 / 0.201 |
| Mask | 53 | 6.3 / 0.200 | 4.6 / 0.170 | 5.4 / 0.190 | 6.1 / 0.201 | 6.4 / 0.214 | 6.3 / 0.214 | 5.8 / 0.200 | 6.0 / 0.201 | 6.3 / 0.213 | 6.4 / 0.212 | 6.6 / 0.220 | 6.3 / 0.199 | 6.4 / 0.208 |
| Metal_Debris | 59 | 7.2 / 0.171 | 4.9 / 0.171 | 5.9 / 0.199 | 7.4 / 0.223 | 7.0 / 0.215 | 6.9 / 0.213 | 7.1 / 0.213 | 7.2 / 0.209 | 6.9 / 0.201 | 7.0 / 0.204 | 7.1 / 0.200 | 7.1 / 0.207 | 6.8 / 0.196 |
| Plastic_Debris | 54 | 6.8 / 0.212 | 4.9 / 0.182 | 5.4 / 0.190 | 6.5 / 0.206 | 6.4 / 0.215 | 7.1 / 0.234 | 6.4 / 0.212 | 6.9 / 0.216 | 6.4 / 0.189 | 6.7 / 0.196 | 6.5 / 0.201 | 6.6 / 0.207 | 6.5 / 0.193 |
| Tire | 69 | 5.6 / 0.141 | 5.1 / 0.171 | 5.6 / 0.188 | 7.0 / 0.201 | 6.9 / 0.205 | 7.1 / 0.208 | 7.0 / 0.206 | 6.9 / 0.203 | 6.9 / 0.194 | 6.8 / 0.189 | 6.4 / 0.188 | 6.8 / 0.188 | 6.8 / 0.191 |

## What is measured in each variant

- The object crops (AUC and diversity per object) include **only the pasted objects**; the kept real boxes of g, i and j (`origin = "real"`) are excluded.
- g, i and j use real TRAIN backgrounds: their **image-level AUC is not comparable** with that of c, f and h (it drops trivially). The key metric is the **object-level AUC**.

| Variant | Background | Images | Pasted objects | Kept real boxes | Rectangular crops discarded | Objects discarded by visibility (rejected attempts) |
|---|---|---|---|---|---|---|
| c_v10_noaug | synthetic | 300 | 531 | 0 | - | - |
| g_v11_copypaste | real (TRAIN) | 300 | 532 | 596 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | - |
| j_v11_visible | real (TRAIN) | 300 | 502 | 595 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (28), Can 3 (25), Fishing_Net 8 (31), Glove 7 (71), Mask 11 (47), Plastic_Debris 2 (19), Tire 1 (30) |
| m_pos | real (TRAIN) | 300 | 494 | 660 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (28), Can 1 (18), Fishing_Net 6 (37), Glove 6 (59), Mask 7 (54), Plastic_Debris 5 (23), Tire 2 (18) |
| p_pos_tight | real (TRAIN) | 300 | 512 | 676 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 1 (17), Can 1 (39), Fishing_Net 5 (33), Glove 7 (74), Mask 5 (65), Plastic_Debris 4 (18) |
| q_base | real (TRAIN) | 300 | 500 | 669 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (39), Can 2 (26), Fishing_Net 6 (34), Glove 7 (66), Mask 8 (42), Plastic_Debris 5 (16) |
| q_lab | real (TRAIN) | 300 | 491 | 677 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (28), Can 6 (79), Fishing_Net 8 (34), Glove 4 (48), Mask 8 (59), Metal_Debris 1 (29), Plastic_Debris 2 (25), Tire 4 (51) |
| q_lab_same | real (TRAIN) | 300 | 493 | 677 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (32), Can 6 (78), Fishing_Net 7 (32), Glove 5 (49), Mask 10 (58), Plastic_Debris 1 (11), Tire 4 (54) |
| q_lab_local | real (TRAIN) | 300 | 493 | 674 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 6 (34), Can 5 (62), Fishing_Net 6 (36), Glove 6 (48), Mask 8 (54), Metal_Debris 1 (22), Plastic_Debris 4 (18), Tire 1 (25) |
| r_matte | real (TRAIN) | 300 | 520 | 677 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 4 (38), Can 9 (96), Fishing_Net 11 (46), Glove 3 (50), Mask 6 (43), Metal_Debris 3 (35), Plastic_Debris 4 (20), Tire 1 (26) |
| r_medium_harm | real (TRAIN) | 300 | 482 | 682 | Bottle 5, Can 1, Fishing_Net 179, Mask 3, Plastic_Debris 1 | Bottle 3 (36), Can 6 (88), Fishing_Net 7 (31), Glove 5 (87), Mask 7 (63), Metal_Debris 1 (37), Plastic_Debris 2 (14), Tire 3 (46) |

## Sharpness, noise, border and scale of the pasted objects (check of the generator improvements)

Medians in the central 50 % of the box at 64 x 64 (hf_ratio, lapvar64) and at native resolution (noise_flat). `border` = fraction of boxes within 3 px of the image border. Reference: real TRAIN objects (n = 21007); the "kept real" objects are the real ones of the same composed images (same pipeline, final JPEG included).

| Variant | Group | n | hf_ratio | lapvar64 | noise_flat | border |
|---|---|---|---|---|---|---|
| TRAIN real | real | 21007 | 0.0315 | 197.91942 | 4.91349 | 0.2914 |
| c_v10_noaug | pasted | 531 | 0.023 | 109.54764 | 4.84762 | 0.6855 |
| g_v11_copypaste | pasted | 532 | 0.01569 | 54.5907 | 3.49367 | 0.5639 |
| g_v11_copypaste | kept real | 596 | 0.028 | 132.0589 | 4.26789 | 0.2466 |
| j_v11_visible | pasted | 502 | 0.01872 | 58.96505 | 4.10843 | 0.5378 |
| j_v11_visible | kept real | 595 | 0.02776 | 150.88315 | 4.70606 | 0.2538 |
| m_pos | pasted | 494 | 0.03135 | 204.48734 | 4.86015 | 0.2874 |
| m_pos | kept real | 659 | 0.02749 | 174.70332 | 5.15708 | 0.2549 |
| p_pos_tight | pasted | 512 | 0.02552 | 85.8526 | 6.10901 | 0.2812 |
| p_pos_tight | kept real | 675 | 0.02652 | 151.09108 | 4.93113 | 0.2607 |
| q_base | pasted | 500 | 0.02044 | 28.91392 | 5.02357 | 0.264 |
| q_base | kept real | 668 | 0.02708 | 165.97146 | 5.00738 | 0.262 |
| q_lab | pasted | 491 | 0.01988 | 29.58305 | 5.15842 | 0.2831 |
| q_lab | kept real | 677 | 0.0271 | 170.33571 | 5.15567 | 0.257 |
| q_lab_same | pasted | 493 | 0.0197 | 30.45926 | 5.11592 | 0.2799 |
| q_lab_same | kept real | 677 | 0.0271 | 170.33571 | 5.1811 | 0.257 |
| q_lab_local | pasted | 493 | 0.02033 | 30.32447 | 4.96916 | 0.2698 |
| q_lab_local | kept real | 673 | 0.02752 | 185.59393 | 5.22311 | 0.2645 |
| r_matte | pasted | 513 | 0.0197 | 32.70491 | 5.711 | 0.2651 |
| r_matte | kept real | 676 | 0.02687 | 160.0924 | 5.0924 | 0.2648 |
| r_medium_harm | pasted | 482 | 0.01964 | 28.65391 | 4.91723 | 0.2759 |
| r_medium_harm | kept real | 682 | 0.02668 | 164.04363 | 5.11211 | 0.261 |

Pasting scale (COCO `scale`; target of `crop_scale_range`: 0.6 - 1.5):

| Variant | n | p10 | p25 | p50 | p75 | p90 | % < 0.5 | % 0.5 - 0.6 | % 0.6 - 1.5 | % > 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| c_v10_noaug | 531 | 0.267 | 0.455 | 0.81 | 1.164 | 1.496 | 27% | 8% | 54% | 10% |
| g_v11_copypaste | 532 | 0.179 | 0.302 | 0.565 | 0.911 | 1.399 | 46% | 7% | 39% | 8% |
| j_v11_visible | 502 | 0.186 | 0.323 | 0.591 | 1.049 | 1.503 | 43% | 7% | 39% | 10% |
| m_pos | 494 | 0.321 | 0.532 | 0.719 | 0.989 | 1.264 | 22% | 11% | 63% | 4% |
| p_pos_tight | 512 | 0.369 | 0.533 | 0.735 | 0.97 | 1.24 | 22% | 10% | 65% | 3% |
| q_base | 500 | 0.375 | 0.559 | 0.734 | 0.979 | 1.273 | 21% | 9% | 67% | 3% |
| q_lab | 491 | 0.313 | 0.492 | 0.679 | 0.908 | 1.243 | 27% | 11% | 58% | 5% |
| q_lab_same | 493 | 0.306 | 0.49 | 0.674 | 0.911 | 1.215 | 27% | 10% | 58% | 4% |
| q_lab_local | 493 | 0.307 | 0.484 | 0.688 | 0.974 | 1.26 | 28% | 10% | 57% | 6% |
| r_matte | 513 | 0.337 | 0.49 | 0.675 | 0.924 | 1.216 | 27% | 11% | 58% | 4% |
| r_medium_harm | 482 | 0.308 | 0.491 | 0.68 | 0.975 | 1.34 | 27% | 11% | 55% | 7% |
