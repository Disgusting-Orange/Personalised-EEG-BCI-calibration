# Stage 4 — Target Quality Report: S001

- Validator version: `stage4-target-quality-1.0`
- Target directory: `outputs\targets\stage3_s001`
- Primary target value: `0.6215415019762847`
- **Verdict: WARNING**

## Traceability

| Check | Passed | Severity | Reason |
|---|---|---|---|
| subject_id_recorded | yes | error | — |
| cross_validation_recorded | yes | error | — |
| decoder_recorded | yes | error | — |
| primary_target_metric_recorded | yes | error | — |
| oof_origin_recorded | yes | error | — |
| fold_metrics_file_present | yes | error | — |
| mi_targets_file_present | yes | error | — |
| software_versions_recorded | yes | warning | — |

## Schema compatibility (downstream stages)

| Check | Passed | Severity | Reason |
|---|---|---|---|
| target_report_top_level_fields | yes | error | — |
| cross_validation_fields | yes | error | — |
| metrics_fields | yes | error | — |
| fold_metrics_columns | yes | error | — |
| mi_targets_columns | yes | error | — |

## Integrity

| Check | Passed | Severity | Reason |
|---|---|---|---|
| target_value_finite | yes | error | — |
| trial_count_coherent | yes | error | — |
| per_class_counts_positive | yes | error | — |
| fold_count_matches_n_splits | yes | error | — |
| fold_balanced_accuracy_finite | yes | error | — |
| mi_targets_subject_present | yes | error | — |

## Reliability metrics

| Metric | Value |
|---|---|
| n_folds | 5.000000 |
| mean | 0.630000 |
| sd | 0.242590 |
| min | 0.325000 |
| max | 1.000000 |

## Quality flags

| Flag | Passed | Value | Threshold | Reason |
|---|---|---|---|---|
| above_chance | yes | 0.6300 | >0.5 | — |
| fold_sd_acceptable | no | 0.2426 | <=0.15 | fold SD 0.2426 exceeds 0.15 |
| sufficient_trials | yes | 45 | >=30 | — |
| class_balance_acceptable | yes | 1.0455 | <=2.0 | — |

## Downstream usability

Target may be used for downstream modelling, but with documented caveats (see failed flags / warnings above).
