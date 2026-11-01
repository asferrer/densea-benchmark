"""Registry of experiments (identifiers are the keys of the result files).

Batch labels: "v9" is the clean-background synthetic batch (default `lot`), "v11" the copy-paste batch;
"v11_r25", "v11_r50" are the copy-paste batches of the 25 % and 50 % real fractions, and "p_pos_tight",
"q_lab_same" the two copy-paste variants of the transfer check.

Fields
  real        fraction of the group-partition train set (class-stratified subset, seed 21; experiments with
              the same percentage share exactly the same real images).
  synth       none | all | minority_focus | inverse_proportional | proxy (a batch built per experiment).
  phase       1 main strategies (the three detectors); 2 incremental series (0/10/25/50/75 % real);
              3 optional experiment (10 % real + IP); 4 transfer check with a 10 % real subset;
              5 strategies with the copy-paste batch; 6 second training seed (22);
              7 incremental series with both batches at 25 % and 50 % real; 8 second seed of the 25 % series;
              9 third training seed (23) of the three 100 % real arms.
  lot         synthetic batch label (see above); `v9` when absent.
  match       (clean-background arms of phase 7) label of the copy-paste batch whose number of pasted
              objects the clean-background sample must match.
  train_seed  training seed, read by the launchers (21 when absent).
  data_from   experiment whose training files are reused (prepare/copy_seed_runs.py); the RF-DETR launcher
              reads prepare/out/rfdetr/<data_from>.
  label       name used in tables and figures.
Builders: prepare_strategies.py (phases 1, 2, 3, 5, 6, 9), prepare_proxy.py (phase 4),
prepare_series.py (phase 7), copy_seed_runs.py (phases 8 and 9 data copies).
"""
EXPERIMENTS = {
    # Phase 1: strategies with 100 % of the real data
    "exp07_real100": {"real": 1.0, "synth": "none", "phase": 1, "label": "Real-Only"},
    "exp06_synth100_real100": {"real": 1.0, "synth": "all", "phase": 1, "label": "Standard Combination"},
    "exp08_real_synth_minor": {"real": 1.0, "synth": "minority_focus", "phase": 1, "label": "Minority-Focus"},
    "exp09_real_synth_invprop": {"real": 1.0, "synth": "inverse_proportional", "phase": 1,
                                 "label": "Inverse-Proportional"},
    # Phase 2: incremental series (0/10/25/50/75 % real, with and without the whole clean-background batch)
    "exp01_synth100": {"real": 0.0, "synth": "all", "phase": 2, "label": "Synthetic only"},
    "exp02_synth100_real10": {"real": 0.10, "synth": "all", "phase": 2, "label": "Synthetic + 10 % real"},
    "exp03_synth100_real25": {"real": 0.25, "synth": "all", "phase": 2, "label": "Synthetic + 25 % real"},
    "exp04_synth100_real50": {"real": 0.50, "synth": "all", "phase": 2, "label": "Synthetic + 50 % real"},
    "exp05_synth100_real75": {"real": 0.75, "synth": "all", "phase": 2, "label": "Synthetic + 75 % real"},
    "exp11_real10": {"real": 0.10, "synth": "none", "phase": 2, "label": "10 % real"},
    "exp12_real25": {"real": 0.25, "synth": "none", "phase": 2, "label": "25 % real"},
    "exp13_real50": {"real": 0.50, "synth": "none", "phase": 2, "label": "50 % real"},
    "exp14_real75": {"real": 0.75, "synth": "none", "phase": 2, "label": "75 % real"},
    # Phase 3 (optional)
    "exp15_real10_synth_invprop": {"real": 0.10, "synth": "inverse_proportional", "phase": 3,
                                   "label": "10 % real + Inverse-Proportional"},
    # Phase 4: transfer check of the generator (VAL only). The same 10 % real subset as exp11 plus 2,000 images
    # of a batch; the copy-paste batches are pasted onto that same 10 % real subset.
    # Built by prepare/prepare_proxy.py.
    "px_real10": {"real": 0.10, "synth": "none", "phase": 4, "label": "Proxy: 10 % real"},
    "px_v9_real10": {"real": 0.10, "synth": "proxy", "lot": "v9", "phase": 4,
                     "label": "Proxy: 10 % real + 2,000 v9"},
    "px_p_real10": {"real": 0.10, "synth": "proxy", "lot": "p_pos_tight", "phase": 4,
                    "label": "Proxy: 10 % real + 2,000 v11 p_pos_tight"},
    "px_q_real10": {"real": 0.10, "synth": "proxy", "lot": "q_lab_same", "phase": 4,
                    "label": "Proxy: 10 % real + 2,000 v11 q_lab_same"},
    # Second training seed of the transfer check (same data), `train_seed` is read by the launchers.
    "px_v9_real10_s2": {"real": 0.10, "synth": "proxy", "lot": "v9", "phase": 4, "train_seed": 22,
                        "label": "Proxy: 10 % real + 2,000 v9 (seed 22)"},
    "px_p_real10_s2": {"real": 0.10, "synth": "proxy", "lot": "p_pos_tight", "phase": 4, "train_seed": 22,
                       "label": "Proxy: 10 % real + 2,000 v11 p_pos_tight (seed 22)"},
    # Phase 5: strategies with the copy-paste batch (12,000 images), VAL only until the final evaluation.
    # MF and IP cover the deficit counting only the pasted objects (prepare/strategies.py).
    # Built by prepare_strategies.py --lot v11. Real-Only is exp07_real100 (independent of the batch).
    "exp06v11_synth100_real100": {"real": 1.0, "synth": "all", "lot": "v11", "phase": 5,
                                  "label": "Standard Combination (v11)"},
    "exp08v11_real_synth_minor": {"real": 1.0, "synth": "minority_focus", "lot": "v11", "phase": 5,
                                  "label": "Minority-Focus (v11)"},
    "exp09v11_real_synth_invprop": {"real": 1.0, "synth": "inverse_proportional", "lot": "v11", "phase": 5,
                                    "label": "Inverse-Proportional (v11)"},
    # Phase 6: second training seed (22) with the same data. VAL only. The data are a copy of
    # prepare/out/<data_from> (only info.description changes).
    "exp07_real100_s2": {"real": 1.0, "synth": "none", "phase": 6, "train_seed": 22, "data_from": "exp07_real100",
                         "label": "Real-Only (seed 22)"},
    "exp08_real_synth_minor_s2": {"real": 1.0, "synth": "minority_focus", "lot": "v9", "phase": 6, "train_seed": 22,
                                  "data_from": "exp08_real_synth_minor",
                                  "label": "Minority-Focus, clean background (seed 22)"},
    "exp06v11_synth100_real100_s2": {"real": 1.0, "synth": "all", "lot": "v11", "phase": 6, "train_seed": 22,
                                     "data_from": "exp06v11_synth100_real100",
                                     "label": "Standard Combination, copy-paste (seed 22)"},
    # Phase 7: incremental series with a constant synthetic-to-real ratio. Same design as phase 4 at 10 %: real
    # fraction of exp12/exp13 (stratified_subset, seed 21); a copy-paste batch per fraction that pastes only onto
    # those images (about 1.17 synthetic images per real image, as 12,000 / 10,245 at 100 %); clean-background
    # arm = random images of the v9 batch until the pasted objects of that copy-paste batch are matched.
    # RT-DETRv2 and VAL only. Built by prepare/prepare_series.py.
    "inc_real25": {"real": 0.25, "synth": "none", "phase": 7, "label": "Series: 25 % real"},
    "inc_v9_real25": {"real": 0.25, "synth": "proxy", "lot": "v9", "match": "v11_r25", "phase": 7,
                      "label": "Series: 25 % real + clean background"},
    "inc_v11_real25": {"real": 0.25, "synth": "proxy", "lot": "v11_r25", "phase": 7,
                       "label": "Series: 25 % real + copy-paste"},
    "inc_real50": {"real": 0.50, "synth": "none", "phase": 7, "label": "Series: 50 % real"},
    "inc_v9_real50": {"real": 0.50, "synth": "proxy", "lot": "v9", "match": "v11_r50", "phase": 7,
                      "label": "Series: 50 % real + clean background"},
    "inc_v11_real50": {"real": 0.50, "synth": "proxy", "lot": "v11_r50", "phase": 7,
                       "label": "Series: 50 % real + copy-paste"},
    # Phase 8: second training seed (22) of the 25 % series, same data (prepare/copy_seed_runs.py).
    "inc_real25_s2": {"real": 0.25, "synth": "none", "phase": 8, "train_seed": 22, "data_from": "inc_real25",
                      "label": "Series: 25 % real (seed 22)"},
    "inc_v9_real25_s2": {"real": 0.25, "synth": "proxy", "lot": "v9", "phase": 8, "train_seed": 22,
                         "data_from": "inc_v9_real25", "label": "Series: 25 % real + clean background (seed 22)"},
    "inc_v11_real25_s2": {"real": 0.25, "synth": "proxy", "lot": "v11_r25", "phase": 8, "train_seed": 22,
                          "data_from": "inc_v11_real25", "label": "Series: 25 % real + copy-paste (seed 22)"},
    # Phase 9: third training seed (23) of the three 100 % real arms, same data (prepare/copy_seed_runs.py).
    # The three detectors also train seed 22 with the phase 6 entries.
    "exp07_real100_s3": {"real": 1.0, "synth": "none", "phase": 9, "train_seed": 23, "data_from": "exp07_real100",
                         "label": "Real-Only (seed 23)"},
    "exp08_real_synth_minor_s3": {"real": 1.0, "synth": "minority_focus", "lot": "v9", "phase": 9, "train_seed": 23,
                                  "data_from": "exp08_real_synth_minor",
                                  "label": "Minority-Focus, clean background (seed 23)"},
    "exp06v11_synth100_real100_s3": {"real": 1.0, "synth": "all", "lot": "v11", "phase": 9, "train_seed": 23,
                                     "data_from": "exp06v11_synth100_real100",
                                     "label": "Standard Combination, copy-paste (seed 23)"},
}
