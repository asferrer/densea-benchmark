"""Static text blocks of the copy-paste gap diagnosis report (summary table and limitations)."""

SUMMARY = """## Summary: main factors (ordered by effect size)

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
"""

LIMITATIONS = """
## Limitations

- 502 pasted objects from 300 images of a single pilot; the AUC of small subsets (n < 50) is trained with few positives and biased downwards
  (e.g. the residual with n = 34). Hanley-McNeil standard errors next to each AUC; the CV seeds vary by ±0.01.
- The regions (interior, border band, surroundings) are geometric rectangles equal for pasted and real objects: real objects have no mask and the generator
  does not trace the rotation angle. "Interior" is the central 50 % of the box, which for a rotated crop includes some background.
- The comparison real objects are 976 TRAIN objects from other images (domain_gap sample; only 101 of Metal_Debris); the 595 kept in the same
  composites have no Metal_Debris and only 10 Can. The class mix is corrected by weighting, not by matching.
- A "factor" here means a measured association (strata and regression; R² = 0.07 on the per-object logit): any intervention has to be checked by
  generating a new pilot and re-running this analysis (`variant_features.py` and `variant_diagnosis.py`).
"""
