# Panel on the final copy-paste batch (v11) against its pilot

Quotas per class (those of gap_panel_drop): {'Bottle': 75, 'Can': 68, 'Fishing_Net': 45, 'Glove': 60, 'Mask': 55, 'Metal_Debris': 64, 'Plastic_Debris': 54, 'Tire': 74}. Pasted objects of the batch: 495.

| Extractor | Set | Grouped AUC [CI] | KID | CMMD | Vendi | Coverage @val | Recall @val |
|---|---|---|---|---|---|---|---|
| resnet50 | pilot p_pos_tight | 0.841 [0.809; 0.886] | 0.0237 | 0.195 | 6.4 | 0.58 | 0.69 |
| resnet50 | pilot s_drop | 0.819 [0.777; 0.870] | 0.0241 | 0.201 | 6.4 | 0.55 | 0.72 |
| resnet50 | full batch | 0.803 [0.783; 0.848] | 0.0190 | 0.157 | 6.6 | 0.59 | 0.75 |
| resnet50 | clean backgrounds (v9) | 0.992 [0.989; 0.996] | 0.1286 | 0.976 | 4.8 | 0.27 | 0.12 |
| resnet50 | real floor (TRAIN vs VAL) | 0.409 [0.339; 0.454] | 0.0025 | 0.036 | 5.8 | – | – |
| dinov2_b14 | pilot p_pos_tight | 0.906 [0.892; 0.930] | 1.2285 | 0.518 | 23.6 | 0.48 | 0.67 |
| dinov2_b14 | pilot s_drop | 0.911 [0.899; 0.933] | 1.2440 | 0.516 | 23.3 | 0.48 | 0.66 |
| dinov2_b14 | full batch | 0.918 [0.906; 0.940] | 1.2291 | 0.506 | 24.0 | 0.43 | 0.70 |
| dinov2_b14 | clean backgrounds (v9) | 0.998 [0.997; 0.999] | 8.3856 | 1.902 | 12.7 | 0.24 | 0.18 |
| dinov2_b14 | real floor (TRAIN vs VAL) | 0.444 [0.373; 0.506] | 0.1211 | 0.055 | 17.4 | – | – |
| siglip2_b16 | pilot p_pos_tight | 0.918 [0.902; 0.940] | 0.0158 | 0.218 | 3.3 | 0.61 | 0.58 |
| siglip2_b16 | pilot s_drop | 0.920 [0.904; 0.948] | 0.0159 | 0.221 | 3.3 | 0.64 | 0.60 |
| siglip2_b16 | full batch | 0.908 [0.894; 0.933] | 0.0134 | 0.188 | 3.5 | 0.64 | 0.68 |
| siglip2_b16 | clean backgrounds (v9) | 0.998 [0.997; 0.999] | 0.0614 | 0.910 | 3.3 | 0.25 | 0.15 |
| siglip2_b16 | real floor (TRAIN vs VAL) | 0.389 [0.316; 0.439] | 0.0012 | 0.027 | 3.5 | – | – |
| dinov3_b16 | pilot p_pos_tight | 0.934 [0.919; 0.952] | 0.0411 | 0.465 | 26.8 | 0.37 | 0.68 |
| dinov3_b16 | pilot s_drop | 0.922 [0.905; 0.945] | 0.0408 | 0.465 | 26.1 | 0.39 | 0.69 |
| dinov3_b16 | full batch | 0.932 [0.916; 0.955] | 0.0383 | 0.439 | 26.7 | 0.39 | 0.75 |
| dinov3_b16 | clean backgrounds (v9) | 1.000 [0.999; 1.000] | 0.1958 | 1.748 | 16.1 | 0.19 | 0.12 |
| dinov3_b16 | real floor (TRAIN vs VAL) | 0.422 [0.359; 0.473] | 0.0029 | 0.059 | 18.0 | – | – |
| clip_b16 | pilot p_pos_tight | 0.909 [0.894; 0.932] | 0.0224 | 0.222 | 3.3 | 0.62 | 0.63 |
| clip_b16 | pilot s_drop | 0.897 [0.875; 0.926] | 0.0231 | 0.230 | 3.3 | 0.66 | 0.66 |
| clip_b16 | full batch | 0.901 [0.879; 0.930] | 0.0205 | 0.207 | 3.5 | 0.63 | 0.71 |
| clip_b16 | clean backgrounds (v9) | 1.000 [1.000; 1.000] | 0.0904 | 0.933 | 3.1 | 0.22 | 0.14 |
| clip_b16 | real floor (TRAIN vs VAL) | 0.412 [0.355; 0.456] | 0.0025 | 0.033 | 3.4 | – | – |

Grouped AUC: 3 CV seeds by partition group; CI by group bootstrap (gap_panel_core). Same quotas in every row; the pilot and the batch come from the same generator (s_drop).
