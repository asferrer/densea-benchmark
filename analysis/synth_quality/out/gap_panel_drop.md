# s_drop vs p_pos_tight (adoption rule)

**Verdict: ADOPT.** Not worse: 5/5; diversity not lower: 5/5 (required minimum 3). Common quotas: {'Bottle': 75, 'Can': 68, 'Fishing_Net': 45, 'Glove': 60, 'Mask': 55, 'Metal_Debris': 64, 'Plastic_Debris': 54, 'Tire': 74}.

| Extractor | AUC control [CI] | AUC s_drop [CI] | dAUC | worse | Vendi c / s | Cov@val c / s | Cov (TRAIN) c / s | KID c / s | CMMD c / s | diversity drop |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 0.841 [0.809; 0.886] | 0.819 [0.777; 0.870] | -0.022 | no | 6.4 / 6.4 | 0.58 / 0.55 | 0.77 / 0.77 | 0.0237 / 0.0241 | 0.195 / 0.201 | no |
| dinov2_b14 | 0.906 [0.892; 0.930] | 0.911 [0.899; 0.933] | +0.005 | no | 23.6 / 23.3 | 0.48 / 0.48 | 0.60 / 0.60 | 1.2285 / 1.2440 | 0.518 / 0.516 | no |
| siglip2_b16 | 0.918 [0.902; 0.940] | 0.920 [0.904; 0.948] | +0.002 | no | 3.3 / 3.3 | 0.61 / 0.64 | 0.66 / 0.70 | 0.0158 / 0.0159 | 0.218 / 0.221 | no |
| dinov3_b16 | 0.934 [0.919; 0.952] | 0.922 [0.905; 0.945] | -0.012 | no | 26.8 / 26.1 | 0.37 / 0.39 | 0.52 / 0.54 | 0.0411 / 0.0408 | 0.465 / 0.465 | no |
| clip_b16 | 0.909 [0.894; 0.932] | 0.897 [0.875; 0.926] | -0.012 | no | 3.3 / 3.3 | 0.62 / 0.66 | 0.67 / 0.67 | 0.0224 / 0.0231 | 0.222 / 0.230 | no |

Rule (set before looking at the numbers):
  * "worse" in an extractor if grouped_AUC(s_drop) - grouped_AUC(control) > 0.02 or AUC(s_drop) exceeds the upper limit
    of the CI (group bootstrap) of the control.
  * "diversity drops" in an extractor if the mean Vendi of s_drop < 0.95 x that of the control, or the coverage with VAL reference
    < coverage of the control - 0.05 (VAL: no shared TRAIN crops). Diversity does not drop if it does not drop in >= 3 of the 5 extractors.
