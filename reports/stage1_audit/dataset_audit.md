# Stage 1 Dataset Audit Report

Date: 2026-07-18
Scope: Stage 1 validation and closure pass only. No preprocessing, feature extraction, target generation, or model training was performed.

## 1. Official documentation summary

Source used: PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 documentation.

Official statements relevant to this audit:

- The dataset contains 14 experimental runs per subject.
- The first two runs are baseline recordings, one with eyes open and one with eyes closed.
- The remaining 12 runs are three repetitions of four task conditions involving real or imagined movement/imagery.
- Event annotations use T0 for rest, T1 for one task class, and T2 for the other task class, but the meaning of T1 and T2 depends on the run context (unilateral fist runs vs bilateral hand/foot runs).
- The baseline recordings should be treated separately from motor-execution and motor-imagery task runs.

## 2. Local observations

Observed from the local copy under data/raw/eegmmidb:

- Subjects found: 109
- EDF recordings found: 1526
- Event sidecar files found: 1526
- Zero-byte files: 0
- Unreadable EDF files: 0
- Duplicate EDF recordings: 0
- Duplicate sidecar/event files: present, but these are not duplicate EEG recordings

## 3. Duplicate validation

The earlier duplicate report was not correctly interpreted.

### What was verified

- No duplicate EDF files were found by content hash.
- The apparent duplicates were repeated .event sidecar files with identical content across different subject directories.

### Conclusion

The duplicate detection logic should be revised so that it does not flag identical event sidecars as duplicate EEG recordings. The correct distinction is:

- Duplicate EDF recordings: none found
- Duplicate sidecar/event files: present, but they are not evidence of duplicate EEG data

## 4. Official run mapping and EO/EC interpretation

### Verified mapping

- R01: baseline recording, eyes open
- R02: baseline recording, eyes closed
- R03-R14: task runs from the four-condition motor execution/imagery protocol

### Important caution

The official documentation does not provide a single universal label for every individual run beyond this protocol-level description. The meaning of T1/T2 is run-context dependent, and the exact mapping of each task block to left/right fist vs both fists/both feet must be handled as a documented run-mapping decision rather than assumed from the file name.

### Local evidence

The local EDF files contain T0/T1/T2 annotations and are consistent with the documented rest/task structure. However, the local files do not carry explicit run names beyond the filename and therefore do not replace the need for an explicit mapping table.

## 5. Known-subject review

The four subjects named in the audit checklist were reviewed again against the official documentation and the local files.

- S088: local evidence shows an anomalous 128 Hz sampling rate for runs R03-R14, while R01-R02 remain at 160 Hz.
- S089: no local 128 Hz sampling-rate anomaly was observed; the concern is the documented event-annotation issue.
- S092: local evidence shows an anomalous 128 Hz sampling rate for runs R03-R14, while R01-R02 remain at 160 Hz.
- S100: local evidence shows an anomalous 128 Hz sampling rate for runs R03-R14, while R01-R02 remain at 160 Hz.

These subjects should be treated as blocked for downstream use until a human approves the handling rule.

## 6. Resting-state validation

The dataset provides baseline recordings with eyes open and eyes closed. They can legitimately be described as resting-state EEG proxies for this project, but they are not a separate long-duration resting-state recording. The limitations are:

- They are one-minute baseline recordings rather than a longer resting-state protocol.
- Eyes-open and eyes-closed baselines are different conditions and should not be assumed interchangeable.
- They are appropriate for methodological exploration of resting-state-related EEG features, but they should be described as baseline resting-state EEG rather than a fully general resting-state dataset.
- The project should treat EO and EC separately unless a specific methodological decision justifies combining them.

This interpretation follows the official dataset description.

## 7. Stage 1 closure note

The dataset audit and methodology validation are complete enough to close this audit stage as a validation pass, but the following items remain pending human review:

- the handling rule for S088, S089, S092, and S100
- the run-level task mapping for R03-R14 beyond the protocol-level official description

No preprocessing, feature extraction, target generation, or model training was started.
