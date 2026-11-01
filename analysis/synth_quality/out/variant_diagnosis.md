# Copy-paste gap diagnosis: what still gives away the pasted objects (j_v11_visible)

TRAIN only. n pasted = 502 (300 composed images over real TRAIN backgrounds); comparison real objects: 976 TRAIN objects (domain_gap sample, recompressed to JPEG q75). CPU only; no VAL or TEST data except the TRAIN vs VAL control quoted in section 0.

## Summary: main factors (ordered by effect size)

Per-object AUC (box with context) of the pasted objects of `j_v11_visible`: **0.85** against real TRAIN objects (0.86 class-adjusted), 0.815 against the
real objects kept in the same composites; the TRAIN vs VAL control is 0.636.

| # | Factor | Evidence | Effect size | Objects affected |
|---|---|---|---|---|
| 1 | **The pasted objects lose fine detail and contrast** (the source crop does not: it already has 0.0286 of hf_ratio and 145 of lapvar64, like the real ones with 0.030 and 170) | sec. 3 and 5: hf_ratio 0.0187 vs 0.0300 (-38 %), lapvar64 58 vs 170 (-66 %), local contrast ΔE 23 vs 36; sharpness family AUC 0.685, contrast 0.642 | the blurriest (< real p25) AUC 0.896 vs 0.826 (+0.070); low contrast +0.028 | 47 % (blurry), 39 % (low contrast) |
| 2 | **It is the object itself and its border, not the surroundings** | sec. 2: interior 0.837, border band 0.875, surroundings 0.747 (0.626 without border boxes, already close to the control 0.636) | interior - surroundings = +0.09 to +0.21 | all 502 |
| 3 | **Position: 54 % of the pasted boxes touch the image border** (real: 28 %, kept 25 %) | sec. 2 and 3: border_dist univariate AUC 0.28; geometry alone AUC 0.737 | AUC 0.850 -> 0.801 when removed (-0.049) | 270 (54 %) |
| 4 | **Extreme scale**: strong reductions (< 0.5) give away more and enlargements 1.2 - 2 the least | sec. 5: scale < 0.5 AUC 0.892 vs 0.834 (+0.058); 1.2 - 2: 0.793 (-0.081); enlarged (> 1.2) with half the noise (2.8 vs 5.0) | without border and with scale 0.5 - 1.5 it drops to 0.740 | 218 (43 %) with scale < 0.5 |
| 5 | Residual not explained by simple statistics | LR on 16 statistics 0.80 < 0.85 of the network; interior and border band stay at 0.82 - 0.83 without border boxes | n/a | all |

**Ruled out** (no measurable effect): crop-background domain match (different vs same: Δ AUC +0.002; 63 % of the pasted objects cross domains), double JPEG
compression (recompressed real 0.850 = not recompressed 0.850; block AUC 0.517), color and saturation (univariate AUC <= 0.534), rectangular crops (none
left), border halo (those with the largest halo are NOT more detectable: Δ -0.03; raw measure) and small crops (Δ +0.005).

## 0. Global AUC per object (box with context)

| Comparison | AUC (grouped CV, 3 seeds) | Class-adjusted AUC |
|---|---|---|
| P vs real TRAIN recompressed q75 (Orc) [main control] | 0.850 ± 0.007 | 0.859 ± 0.011 |
| P vs original real TRAIN (Oplain) [variant_pilot recipe] | 0.850 ± 0.006 | 0.860 ± 0.011 |
| P (without Metal_Debris) vs real kept in the same composites (RC, n=595) | 0.815 ± 0.005 | 0.803 ± 0.017 |

Control real TRAIN vs VAL (domain_gap, already measured): 0.636. n pasted = 502, n real (Orc) = 976.

## 1. AUC per class

"Global model": AUC inside the class of the out-of-fold probabilities of the 8-class classifier (± Hanley-McNeil standard error). "Per-class model": classifier trained only with that class.

| Class | n pasted | n real | Global-model AUC | Per-class-model AUC |
|---|---|---|---|---|
| Bottle | 74 | 125 | 0.847 ± 0.031 | 0.804 ± 0.028 |
| Can | 74 | 125 | 0.865 ± 0.029 | 0.933 ± 0.007 |
| Fishing_Net | 28 | 125 | 0.863 ± 0.046 | 0.946 ± 0.006 |
| Glove | 60 | 125 | 0.876 ± 0.031 | 0.949 ± 0.005 |
| Mask | 63 | 125 | 0.853 ± 0.032 | 0.899 ± 0.009 |
| Metal_Debris | 71 | 101 | 0.897 ± 0.027 | 0.939 ± 0.010 |
| Plastic_Debris | 60 | 125 | 0.805 ± 0.037 | 0.765 ± 0.017 |
| Tire | 72 | 125 | 0.876 ± 0.028 | 0.950 ± 0.010 |

## 2. AUC per region

Same geometric regions in pasted and real objects (gray fill outside the region). The interior is the central rectangle (not the mask, which real objects do not have); in the surroundings the whole box is covered.

| Region | AUC | Class-adjusted AUC | Δ vs context | AUC without boxes at the image border |
|---|---|---|---|---|
| (a) box with context | 0.850 ± 0.007 | 0.859 ± 0.011 | - | 0.801 ± 0.008 |
| (b) interior only (central 50 %) | 0.837 ± 0.007 | 0.840 ± 0.012 | -0.013 | 0.819 ± 0.006 |
| (c) box border band only | 0.875 ± 0.001 | 0.884 ± 0.010 | +0.025 | 0.827 ± 0.008 |
| (d) surroundings only (without the box) | 0.747 ± 0.003 | 0.760 ± 0.014 | -0.103 | 0.626 ± 0.011 |

Pasted boxes at the image border (within 3 px): 270 of 502 pasted (54 %) vs 273 of 976 real TRAIN (28 %) and 25 % of the real objects kept in the same composites. The last column uses only objects that do not touch the border (n pasted = 232, n real = 703).

## 3. Low-level statistics (interior region, central 50 % of the box)

Medians: P pasted, RC real of the same composites, Orc real TRAIN recompressed q75, Oplain not recompressed. Univariate AUC (pasted = positive; 0.5 = does not separate; class-adjusted) against Orc. `noise_flat`, `lapvar_native` and `block` depend on the native resolution (CleanSea 1920x1080, Roboflow 416x416); `lapvar64` and `hf_ratio` are measured at 64x64.

| Statistic | P | RC | Orc | Oplain | Univ. AUC | |AUC-0.5| |
|---|---|---|---|---|---|---|
| lapvar_native | 167.5 | 359.8 | 421.3 | 421.3 | 0.386 ± 0.015 | 0.114 |
| lapvar64 | 57.95 | 148.9 | 170.1 | 170.1 | 0.383 ± 0.015 | 0.117 |
| hf_ratio | 0.01872 | 0.02773 | 0.03004 | 0.03004 | 0.343 ± 0.015 | 0.157 |
| noise_flat | 4.427 | 4.711 | 5.166 | 5.183 | 0.456 ± 0.016 | 0.044 |
| block | 1.291 | 1.321 | 1.267 | 1.267 | 0.517 ± 0.016 | 0.017 |
| sat | 138.6 | 150.2 | 133.1 | 133.1 | 0.534 ± 0.016 | 0.034 |
| val | 132.2 | 134.7 | 138.4 | 138.4 | 0.462 ± 0.016 | 0.038 |
| hue_cos | -0.8244 | -0.8501 | -0.7367 | -0.7367 | 0.478 ± 0.016 | 0.022 |
| hue_sin | 0.0756 | -0.02901 | 0.1155 | 0.116 | 0.488 ± 0.016 | 0.012 |
| contrast_dE | 23.39 | 33.95 | 36.22 | 36.22 | 0.366 ± 0.015 | 0.134 |
| l_std | 30.47 | 33.64 | 32.28 | 32.28 | 0.471 ± 0.016 | 0.029 |
| rel_size | 0.2364 | 0.2835 | 0.2632 | 0.2632 | 0.490 ± 0.016 | 0.010 |
| log_aspect | 0.00981 | -0.1112 | 0.01189 | 0.01189 | 0.499 ± 0.016 | 0.001 |
| cx | 0.4706 | 0.506 | 0.5048 | 0.5048 | 0.486 ± 0.016 | 0.014 |
| cy | 0.5739 | 0.5024 | 0.5102 | 0.5102 | 0.542 ± 0.016 | 0.042 |
| border_dist | 0.002404 | 0.09135 | 0.08237 | 0.08237 | 0.280 ± 0.013 | 0.220 |
| min_side | 93.5 | 98 | 104 | 104 | 0.468 ± 0.016 | 0.032 |

AUC of an LR classifier on each family (grouped CV):

| Family | AUC |
|---|---|
| sharpness (lapvar_native, lapvar64, hf_ratio) | 0.685 ± 0.001 |
| noise in flat regions (noise_flat) | 0.556 ± 0.001 |
| JPEG blocks (block) | 0.491 ± 0.009 |
| color (sat, val, hue_cos, hue_sin) | 0.515 ± 0.006 |
| contrast (contrast_dE, l_std) | 0.642 ± 0.001 |
| geometry (rel_size, log_aspect, cx, cy, border_dist, min_side) | 0.737 ± 0.002 |
| ALL the statistics together | 0.798 ± 0.003 |

## 4. Domain of the crop's origin vs domain of the target image

AUC of the pasted objects of each cell against the real objects of the SAME target domain (class-adjusted, out-of-fold probabilities of model 0). Cells with n < 15 are noisy.

| Crop domain | Target domain | Relation | n | AUC | median logit |
|---|---|---|---|---|---|
| CleanSea | CleanSea | same | 8 | 0.870 ± 0.082 | +1.65 |
| CleanSea | Neural Ocean | different | 37 | 0.852 ± 0.043 | +1.07 |
| CleanSea | Ocean Garbage | different | 80 | 0.840 ± 0.031 | +1.15 |
| Neural Ocean | CleanSea | different | 12 | 0.700 ± 0.091 | -1.15 |
| Neural Ocean | Neural Ocean | same | 64 | 0.820 ± 0.034 | +0.75 |
| Neural Ocean | Ocean Garbage | different | 98 | 0.897 ± 0.022 | +1.65 |
| Ocean Garbage | CleanSea | different | 15 | 0.782 ± 0.079 | -0.03 |
| Ocean Garbage | Neural Ocean | different | 73 | 0.831 ± 0.031 | +0.63 |
| Ocean Garbage | Ocean Garbage | same | 115 | 0.872 ± 0.022 | +1.31 |

Aggregates (AUC inside / outside the criterion against all the real objects):

| Criterion | n | % pasted | AUC inside | AUC outside | Δ | median logit inside/outside |
|---|---|---|---|---|---|---|
| crop domain = target domain | 187 | 37 % | 0.858 ± 0.018 | 0.860 ± 0.014 | -0.002 | +1.22 / +1.11 |
| crop domain != target domain | 315 | 63 % | 0.860 ± 0.014 | 0.858 ± 0.018 | +0.002 | +1.11 / +1.22 |
| Roboflow crop -> CleanSea target | 27 | 5 % | 0.715 ± 0.057 | 0.867 ± 0.011 | -0.152 | -0.28 / +1.28 |
| CleanSea crop -> Roboflow target | 117 | 23 % | 0.873 ± 0.022 | 0.855 ± 0.013 | +0.018 | +1.14 / +1.11 |
| Roboflow crop -> Roboflow target (different subdomain) | 171 | 34 % | 0.874 ± 0.018 | 0.851 ± 0.014 | +0.023 | +1.32 / +0.97 |

## 5. Crop quality and scaling

AUC of the pasted objects that meet the criterion against the real objects with the same class mix, and of those that do not (model 0). Δ > 0: the criterion makes them MORE detectable.

| Criterion | n pasted affected | % pasted | AUC inside | AUC outside | Δ | median logit inside/outside |
|---|---|---|---|---|---|---|
| blurrier than the real p25 (hf_ratio < 0.0180) | 236 | 47 % | 0.896 ± 0.014 | 0.826 ± 0.016 | +0.070 | +1.62 / +0.60 |
| low local contrast, < real p25 (contrast_dE < 18) | 197 | 39 % | 0.876 ± 0.017 | 0.848 ± 0.015 | +0.028 | +1.47 / +0.80 |
| box at the image border (<= 3 px) | 270 | 54 % | 0.868 ± 0.015 | 0.849 ± 0.017 | +0.019 | +1.09 / +1.13 |
| scale > 1.0 (enlarged crop) | 135 | 27 % | 0.825 ± 0.023 | 0.872 ± 0.013 | -0.047 | +0.63 / +1.28 |
| scale > 1.5 | 51 | 10 % | 0.810 ± 0.037 | 0.865 ± 0.012 | -0.055 | +0.62 / +1.14 |
| scale < 0.5 (strongly reduced) | 218 | 43 % | 0.892 ± 0.015 | 0.834 ± 0.016 | +0.058 | +1.47 / +0.68 |
| original crop < 40 px on its smaller side | 52 | 10 % | 0.864 ± 0.033 | 0.859 ± 0.012 | +0.005 | +1.38 / +1.09 |
| effective resolution (smaller side x scale) < 40 px | 153 | 30 % | 0.875 ± 0.019 | 0.852 ± 0.014 | +0.023 | +1.30 / +0.99 |
| rectangular crop (alpha covers > 95 %): original background included | 0 | 0 % | n/a | 0.859 ± 0.011 | n/a | - |
| border halo > median (0.65) | 251 | 50 % | 0.847 ± 0.016 | 0.872 ± 0.015 | -0.025 | +0.89 / +1.38 |
| border halo > 0.5 (border ~ original background color) | 320 | 64 % | 0.848 ± 0.014 | 0.879 ± 0.017 | -0.031 | +0.98 / +1.38 |
| original-background vs target color mismatch > median (89) | 251 | 50 % | 0.861 ± 0.015 | 0.857 ± 0.016 | +0.005 | +1.22 / +1.03 |
| relative target size > p75 | 126 | 25 % | 0.824 ± 0.023 | 0.871 ± 0.012 | -0.047 | +0.44 / +1.29 |
| relative target size < p25 | 126 | 25 % | 0.872 ± 0.021 | 0.855 ± 0.013 | +0.018 | +1.29 / +1.05 |
| CleanSea target (1920x1080, video) | 35 | 7 % | 0.747 ± 0.049 | 0.868 ± 0.011 | -0.120 | -0.03 / +1.22 |

By scale-factor range:

| Scale | n | % pasted | AUC inside | AUC outside | Δ | median logit |
|---|---|---|---|---|---|---|
| < 0.5 | 218 | 43 % | 0.892 ± 0.015 | 0.834 ± 0.016 | +0.058 | +1.47 / +0.68 |
| 0.5 - 0.8 | 92 | 18 % | 0.834 ± 0.027 | 0.865 ± 0.012 | -0.030 | +0.84 / +1.14 |
| 0.8 - 1.2 | 100 | 20 % | 0.871 ± 0.023 | 0.856 ± 0.013 | +0.015 | +0.99 / +1.14 |
| 1.2 - 2 | 92 | 18 % | 0.793 ± 0.029 | 0.874 ± 0.012 | -0.081 | +0.32 / +1.29 |
| >= 2 | 0 | 0 % | n/a | 0.859 ± 0.011 | n/a | - |

Sharpness and contrast per scale range (medians):

| Group | hf_ratio | lapvar64 | noise_flat | contrast_dE | l_std | n |
|---|---|---|---|---|---|---|
| real Orc | 0.03004 | 170.1 | 5.166 | 36.22 | 32.28 | 976 |
| pasted scale < 0.5 | 0.01641 | 37.43 | 4.994 | 19.16 | 29.9 | 218 |
| pasted 0.5 - 1.2 | 0.01895 | 82.75 | 4.957 | 25.57 | 30.98 | 192 |
| pasted >= 1.2 | 0.02256 | 75.41 | 2.825 | 27.64 | 29.45 | 92 |

Is the sharpness loss due to the crop or to the generator? (statistics at 64x64)

The "emulation" rows replicate the deterministic stages of `_transform_object` on the crop of each pasted object (random rotation, recorded scale); the rest of the drop down to the row "pasted in the composite" comes from harmonization, blur matching to the background, lighting, caustics/veil, alpha composition and JPEG q75.

| What is measured | median hf_ratio | median lapvar64 |
|---|---|---|
| source crop (PNG, central 50 %, before pasting) | 0.0286 | 144.8 |
| emulation: + Gaussian pre-blur (crops < 100 px) | 0.0271 | 106.3 |
| emulation: + rotation ±45° (cubic) | 0.0297 | 124.2 |
| emulation: + scale (AREA / LANCZOS) | 0.0217 | 32.7 |
| pasted in the composite (same region, after scale, rotation, blur, harmonization and JPEG) | 0.0187 | 57.9 |
| real TRAIN (Orc) | 0.0300 | 170.1 |
| median composite / source ratio (per object) | 0.67 | 0.55 |

Residual AUC (box with context) after removing the giveaway cases:

| Subset | n pasted | n real | AUC | Class-adjusted AUC |
|---|---|---|---|---|
| without boxes pasted at the border | 232 | 703 | 0.801 ± 0.008 | 0.816 ± 0.018 |
| without border and scale 0.5 - 1.5 | 99 | 703 | 0.740 ± 0.005 | 0.752 ± 0.030 |
| without border, scale 0.5 - 1.5, sharpness and contrast >= real p25 | 34 | 703 | 0.582 ± 0.013 | 0.588 ± 0.053 |
| only sharpness and contrast >= real p25 (border and scale untouched) | 167 | 976 | 0.785 ± 0.010 | 0.796 ± 0.021 |

What explains the detectability of each pasted object (logit of the out-of-fold score of model 0; n = 502; R² = 0.07):

| Covariate | Spearman | standardized coef. | t |
|---|---|---|---|
| log scale | -0.12 | -0.20 | -2.1 |
| log smaller crop side | -0.04 | -0.15 | -1.6 |
| halo | -0.05 | -0.11 | -2.4 |
| background color mismatch | +0.09 | +0.09 | +1.9 |
| relative target size | -0.09 | +0.05 | +0.7 |
| touches the border | +0.02 | +0.04 | +0.9 |
| cross domain | +0.00 | -0.01 | -0.2 |
| hf_ratio | -0.18 | -0.12 | -2.3 |
| noise_flat | -0.05 | +0.01 | +0.2 |
| contrast_dE | -0.08 | +0.01 | +0.2 |
| block | +0.06 | -0.01 | -0.2 |
| sat | +0.15 | +0.11 | +2.1 |


## Limitations

- 502 pasted objects from 300 images of a single pilot; the AUC of small subsets (n < 50) is trained with few positives and biased downwards
  (e.g. the residual with n = 34). Hanley-McNeil standard errors next to each AUC; the CV seeds vary by ±0.01.
- The regions (interior, border band, surroundings) are geometric rectangles equal for pasted and real objects: real objects have no mask and the generator
  does not trace the rotation angle. "Interior" is the central 50 % of the box, which for a rotated crop includes some background.
- The comparison real objects are 976 TRAIN objects from other images (domain_gap sample; only 101 of Metal_Debris); the 595 kept in the same
  composites have no Metal_Debris and only 10 Can. The class mix is corrected by weighting, not by matching.
- A "factor" here means a measured association (strata and regression; R² = 0.07 on the per-object logit): any intervention has to be checked by
  generating a new pilot and re-running this analysis (`variant_features.py` and `variant_diagnosis.py`).
