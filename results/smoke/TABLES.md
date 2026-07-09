# Results — contrast: presence

**Verdict:** On 'presence' (n=4 held-out subjects): SSL-pretrained Model B does NOT outperform the single-subject baseline at the 25% budget (median Δbalanced-acc=-0.486, Wilcoxon p=1, not significant at α=0.05). Data-efficiency: ~1.0x less target data to match the from-scratch EEGNet baseline.

## T1 — Data-efficiency curve (headline)

| budget_pct | A1_matched | A2_eegnet | B_finetune | B_linprobe | delta_B_A2 | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 0.766±0.235 | 1.000±0.000 | 0.523±0.026 | 0.500±0.000 | -0.477 | 1.0 | 1.0 |
| 100 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.500±0.000 | 0.0 | 1.0 | 1.0 |

## T2 — Per-subject @ 25% budget

| hidden_subject | A1_matched | A2_eegnet | B_finetune | delta_B_A2 |
| --- | --- | --- | --- | --- |
| sub-s00 | 0.5 | 1.0 | 0.5 | -0.5 |
| sub-s01 | 1.0 | 1.0 | 0.528 | -0.472 |
| sub-s02 | 0.562 | 1.0 | 0.562 | -0.438 |
| sub-s03 | 1.0 | 1.0 | 0.5 | -0.5 |

## T3 — Pretraining-breadth ablation

| study_pool_size | B_finetune_mean | B_finetune_sd | delta_vs_A2_full | n_subjects |
| --- | --- | --- | --- | --- |
| 1.0 | 0.852 | 0.207 | -0.148 | 4.0 |

## T4 — Label provenance

| subject | contrast | provisional | n_valid | class_counts | pos_frac |
| --- | --- | --- | --- | --- | --- |
| sub-s00 | presence | False | 120 | {0: 64, 1: 56} | 0.467 |
| sub-s01 | presence | False | 120 | {1: 61, 0: 59} | 0.508 |
| sub-s02 | presence | False | 120 | {1: 66, 0: 54} | 0.55 |
| sub-s03 | presence | False | 120 | {0: 68, 1: 52} | 0.433 |
