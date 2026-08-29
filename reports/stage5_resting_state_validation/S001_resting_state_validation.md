# Stage 5 — Resting-State Validation Report: S001

- Validator version: `stage5-resting-state-1.0`
- **Verdict: PASS**
- Baseline runs validated: 2

## S001/R01 (eyes_open)

### Authoritative FIF metadata

| Property | Value |
|---|---|
| sfreq | 128.0 |
| n_channels | 64 |
| ch_names | 64 channels |
| n_eeg_channels | 64 |
| highpass | 1.0 |
| lowpass | 40.0 |
| file_path | outputs\preprocessed\filtered\S001_R01_filtered_raw.fif |

### Validation checks

| Check | Passed | Severity | Reason |
|---|---|---|---|
| filtered_fif_exists | yes | error | — |
| epochs_fif_exists | yes | error | — |
| ica_fif_exists | yes | error | — |
| qc_report_exists | yes | error | — |
| sfreq_matches_config | yes | error | — |
| n_channels_expected | yes | error | — |
| eeg_channels_present | yes | error | — |
| filter_bandpass_recorded | yes | warning | highpass=1.0, lowpass=40.0 |
| qc_sfreq_consistent_with_fif | yes | warning | — |
| usable_epochs_sufficient | yes | warning | — |
| rejection_rate_acceptable | yes | warning | — |
| bad_channels_acceptable | yes | warning | — |
| ica_components_recorded | yes | info | — |
| epochs_sfreq_matches_config | yes | error | — |

## S001/R02 (eyes_closed)

### Authoritative FIF metadata

| Property | Value |
|---|---|
| sfreq | 128.0 |
| n_channels | 64 |
| ch_names | 64 channels |
| n_eeg_channels | 64 |
| highpass | 1.0 |
| lowpass | 40.0 |
| file_path | outputs\preprocessed\filtered\S001_R02_filtered_raw.fif |

### Validation checks

| Check | Passed | Severity | Reason |
|---|---|---|---|
| filtered_fif_exists | yes | error | — |
| epochs_fif_exists | yes | error | — |
| ica_fif_exists | yes | error | — |
| qc_report_exists | yes | error | — |
| sfreq_matches_config | yes | error | — |
| n_channels_expected | yes | error | — |
| eeg_channels_present | yes | error | — |
| filter_bandpass_recorded | yes | warning | highpass=1.0, lowpass=40.0 |
| qc_sfreq_consistent_with_fif | yes | warning | — |
| usable_epochs_sufficient | yes | warning | — |
| rejection_rate_acceptable | yes | warning | — |
| bad_channels_acceptable | yes | warning | — |
| ica_components_recorded | yes | info | — |
| epochs_sfreq_matches_config | yes | error | — |

## Cross-run comparison (EO vs EC)

| Check | Passed | Severity | Reason |
|---|---|---|---|
| both_conditions_present | yes | error | — |
| channel_names_consistent | yes | error | — |
| sfreq_consistent_across_runs | yes | error | — |

## Known Stage 2 reporting issues

The following inconsistencies in Stage 2 output are documented here. They are NOT repaired (Stage 2 is frozen). Filtered FIF metadata is treated as authoritative.

### sampling_rate_qc_field_conflict

> The QC report sampling_frequency_hz field may not match the actual filtered FIF file's sampling frequency. Per Stage 5 policy, the FIF metadata is treated as authoritative.


- Affected runs: S001/R01
- Authoritative source: filtered FIF sfreq metadata

### epoch_rejection_logging_inconsistency

> The R01 epoch pipeline log reports all 30 epochs dropped, yet the R01 QC report shows n_usable_epochs=30 and the epoch FIF contains 30 epochs. The R02 execution shows 0 epochs dropped with identical n_usable_epochs=30.


- Affected runs: S001/R01
- Authoritative source: epoch FIF contents (n_epochs)

### validation_log_gap

> R02 entries were not appended to the legacy *_validation.log files (filter_validation.log, etc.) because the original R01 entries were written by an external harness using a 'validation' logger that is not part of the frozen src/preprocessing/ code.


- Affected runs: S001/R02
- Authoritative source: console output from Stage 2 pipeline functions

## Methodological evaluation (AGENTS.md §8)

Per AGENTS.md Section 8, the following evaluates whether EEGMMIDB baseline recordings are appropriate as resting-state EEG input.

### What EEGMMIDB provides

The PhysioNet EEGMMIDB provides two 1-minute baseline recordings per subject: R01 (eyes open) and R02 (eyes closed), recorded at 64 channels with 160 Hz native sampling rate. These are the ONLY baseline/rest recordings in the dataset. There is no separate dedicated resting-state recording (e.g., 5 minutes of rest).

### Eyes-open (R01) appropriateness

Eyes-open resting-state EEG is a widely used condition in neuroscientific research. It captures visual-alpha desynchronisation and is suitable for functional connectivity analysis. However, EEGMMIDB's 1-minute R01 recording is short relative to typical resting-state protocols (which often use 3-10 minutes). This limits spectral resolution, particularly in lower-frequency bands (delta, theta).

### Eyes-closed (R02) appropriateness

Eyes-closed resting-state EEG is the classical condition for alpha-band analysis and is standard in connectivity studies. The 1-minute R02 recording has the same duration limitation as R01. Eyes-closed may yield stronger alpha oscillations, which can benefit alpha-band connectivity estimates.

### Separate vs combined analysis

EO and EC are physiologically distinct conditions with different spectral profiles (alpha power is typically higher in EC). Combining them would mix two different neural states and could obscure condition-specific connectivity patterns. They should be analysed separately as the primary approach, with combined analysis as a secondary sensitivity test only if scientifically justified.

### Limitations

Several limitations must be reported: (1) Duration: ~1 minute per condition is short for reliable resting-state connectivity estimation. This constrains spectral resolution and frequency-band specificity. (2) Task context: these recordings were collected as baselines within a motor-task experiment, not as dedicated resting-state sessions. The subject's state may differ from a pure resting-state protocol. (3) No pre-task vs post-task distinction: it is unknown whether R01/R02 precede or follow the motor-task runs, so fatigue or task-aftereffects cannot be ruled out. (4) Only one recording per condition per subject provides no test-retest reliability within the dataset. These limitations must be explicitly reported in any publication using these recordings as resting-state input.

### WBS Journal-track requirement

Per the WBS Journal track (task J2.2), EO vs EC must be tested separately as a robustness analysis regardless of which is used as the primary resting-state input. This Stage 5 validation enables that comparison by ensuring both conditions have validated preprocessing outputs.

## Downstream usability

Resting-state preprocessing outputs are validated and suitable for downstream spectral feature extraction and connectivity analysis.
