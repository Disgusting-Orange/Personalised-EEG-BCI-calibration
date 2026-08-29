# AGENTS.md — Personalised EEG BCI Calibration V2

## 1. Project Purpose

This repository contains a fresh, publication-oriented implementation of the research project:

**Personalised EEG BCI Calibration via Resting-State Connectivity Prediction**

The central research question is:

> Can subject-level resting-state EEG characteristics, particularly spectral and functional connectivity representations, predict an individual's continuous motor-imagery (MI) BCI decoding performance?

The project uses the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), expected to contain 109 subjects, 64 channels, 160 Hz.

The raw dataset is located under:

`data/raw/eegmmidb/`

The project Work Breakdown Structure (WBS) is located at:

`docs/MI_BCI_WBS_.xlsx`

The WBS is the base project specification, is supervisor-approved, and defines the actual submission tracks, milestones, and quality gates. It should be inspected before implementing the research pipeline.

The methodological rules and regression override defined in this AGENTS.md take precedence where the WBS specifies binary good/poor performer classification as the *primary* framing (see Section 2). They do not override the WBS's tracks, milestones, or supervisor sign-off requirements (see Section 4a). The conflicts previously logged here have been resolved by explicit human decision (Section 1a) and by remaining open items still requiring downstream review (Section 1b).

---

## 1a. Resolved Decisions (human-approved, superseding earlier flagged conflicts)

These decisions were made explicitly by the project owner and apply throughout this file:

1. **GCN vs. GAT primacy — resolved.** GCN is the primary graph model for the core/EMBC pipeline. GAT is a secondary comparison/ablation model scoped to the extended Journal track, consistent with the WBS. GAT may only be promoted to a co-primary model by a later explicit decision — the agent must not do this unilaterally.
2. **AutoML — resolved.** AutoML is not a mandatory V2 stage or WBS milestone. It is optional exploratory analysis only, and must never delay or replace any WBS-required experiment or milestone.
3. **Classification-framed WBS Quality Gates — resolved.** Continuous regression is the primary V2 analysis. No classification-framed gate is silently skipped; each maps to an explicit regression equivalent, documented in Section 2a. The final paired regression model-comparison test (in place of McNemar's test) is not hard-coded — it is flagged for statistical review before statistical testing begins.
4. **Known-issue subjects (S088, S089, S092, S100) — resolved as a procedure, not an outcome.** These subjects must be audited, not automatically excluded or modified. See Section 6a for the required procedure.
5. **WBS preprocessing defaults — resolved as adopted starting parameters.** See Section 9a for the constraints on how they may be adjusted.
6. **Both tracks and their milestone structure are preserved as-is** — see Section 4a for the three-way distinction between technical completion, quality-gate completion, and supervisor sign-off.

---

## 1b. Remaining Open Items for Downstream Review

Not conflicts between this file and the WBS — items intentionally deferred rather than decided now:

1. **Final paired statistical test for regression model comparison** (replacing McNemar's test in Section 2a) is not yet chosen. A paired statistical comparison based on appropriate paired out-of-fold predictions or errors will be selected and justified before statistical analysis. Candidate methods include paired permutation testing or Wilcoxon signed-rank testing where their assumptions are satisfied — neither is assumed in advance.
2. **Inclusion/exclusion/correction rules for S088, S089, S092, S100** are not yet defined — Section 6a requires the audit stage to *propose* pre-registered rules, which then require explicit human approval before any downstream stage relies on these subjects.

---

## 2. Critical Methodological Override: Continuous Regression

The primary subject-level prediction problem MUST be formulated as **continuous regression**, not binary good-vs-poor performer classification.

A previous experimental implementation converted MI decoding performance into binary labels using an approximately 70% accuracy threshold:

- Poor performer
- Good performer

This threshold-based classification is NOT the primary methodology for this V2 implementation.

The new primary pipeline must preserve each subject's continuous MI decoding performance.

Conceptually:

Raw MI EEG
→ Preprocessing
→ MI task decoding
→ Cross-validated subject-level MI decoding performance
→ Continuous target

The primary continuous target is **balanced accuracy** from the cross-validated MI decoder. It is a continuous subject-level score, not a binary class label.

The downstream research problem is:

Resting-state EEG representation
→ Regression model
→ Predicted continuous MI decoding performance

Do NOT arbitrarily convert continuous MI performance into binary labels for the primary analysis.

Binary good/poor performer classification may only be included later as a secondary or supplementary sensitivity analysis if a scientifically defensible threshold is established — this matches the WBS's own Journal-track framing of the 60/70/80% threshold test (task J2.4) as a *robustness/sensitivity* analysis, not a primary result.

### 2a. Reconciling the WBS Quality Gates sheet with this override

The WBS's "✅ Quality Gates" sheet is written in classification language and is a required pre-submission checklist for both tracks. Since this file's regression override takes precedence, each classification-framed gate maps to a regression equivalent as follows. The agent must satisfy the regression-equivalent version; do not satisfy the letter of the WBS gate by reporting classification metrics as the primary result.

| WBS Quality Gate (as written) | Regression-override equivalent to satisfy instead |
|---|---|
| "Balanced Accuracy used as primary metric" | Balanced accuracy is the continuous MI target metric. MAE, RMSE, and R² are the primary metrics for the later subject-level regression models (per Section 14); Pearson/Spearman correlation reported where appropriate |
| "Good/poor performer threshold defined before model training; class balance reported" | No threshold is defined for the primary pipeline. If a supplementary binary sensitivity analysis is run later (per Section 2 above), the threshold and resulting class balance are reported only in that supplementary section |
| "McNemar's or Wilcoxon test: GCN vs best ML model" | McNemar's test does not apply to a regression comparison. A paired statistical comparison based on appropriate paired out-of-fold predictions or errors will be selected and justified before statistical analysis. Candidate methods include paired permutation testing or Wilcoxon signed-rank testing where their assumptions are satisfied. Note: with LOSO, a single outer fold holds out one subject, so a fold-level *correlation* is not defined — any paired comparison must operate on per-subject predictions or errors, not per-fold correlations. The specific test is **not hard-coded** — it is flagged for statistical review before Stage 13 (statistical testing) begins, per Section 1b |
| "All 4 ML baselines (RF, XGBoost, SVM, Ridge) included and reported" | Unchanged — still required, but reported with regression metrics, plus a DummyRegressor/mean-predictor baseline per Section 13 |
| "Permutation test (n=1000) for every model — p-value confirms above-chance" | Unchanged in spirit — run permutation testing on the regression target per Section 19, not on binary labels |
| Everything else on the Quality Gates sheet (data/reproducibility, interpretability, writing, ethics rows) | Unchanged — these are not classification-specific and apply as written |

---

## 3. High-Level Research Pipeline

The intended pipeline is:

Raw EEGMMIDB
→ Dataset validation and integrity audit
→ EEG preprocessing
→ Quality control
→ Separation of relevant resting-state and motor-imagery recordings

Motor-imagery branch:
MI EEG
→ Correct task/event extraction
→ MI decoding pipeline
→ Leakage-safe within-subject cross-validation
→ Continuous per-subject MI decoding performance

Resting-state branch:
Resting-state EEG
→ Spectral features
→ Functional connectivity
→ Subject-level feature representations
→ Graph representations

Prediction branch:
Resting-state representation
→ Classical regression baselines
→ XGBoost regression
→ AutoML regression (optional/exploratory only — see Section 1a; must not delay or replace required experiments)
→ GCN regression (primary graph model, core/EMBC pipeline)
→ GAT regression (secondary comparison/ablation model, Journal track only — see Section 1a)

Evaluation:
→ Subject-independent validation
→ MAE
→ RMSE
→ R²
→ Pearson correlation where appropriate
→ Spearman correlation where appropriate
→ Confidence intervals
→ Permutation testing
→ Ablation studies

Final:
→ Statistical analysis
→ Interpretability
→ Publication-quality figures
→ Reproducibility package

---

## 4. Mandatory Stage-by-Stage Development

Do NOT implement the complete pipeline in one autonomous run.

Work must proceed through explicit stages.

The stages are:

1. Dataset audit and run-mapping validation — complete and frozen.
2. Preprocessing and preprocessing QC — complete, reviewed, validated, and frozen.
3. MI event/task extraction, CSP + LDA decoding, and continuous target generation — current stage; validate on S001 only using stratified 5-fold cross-validation and balanced accuracy.
4. Target quality and reliability analysis
5. Resting-state preprocessing validation
6. Resting-state spectral feature extraction
7. Functional connectivity extraction
8. Classical regression baselines
9. XGBoost regression
10. Graph dataset construction
11. GCN regression (primary — core/EMBC pipeline)
12. GAT regression (secondary comparison/ablation — Journal track only, Section 1a)
13. Statistical testing
14. Ablation studies
15. Interpretability
16. Publication figures
17. Reproducibility audit

The agent must stop at the end of the explicitly requested stage.

Do NOT automatically proceed to later stages unless instructed.

### 4a. Relationship to the WBS tracks and milestones

The WBS defines two supervisor-approved tracks layered on top of these 21 stages:

- **EMBC track** (~16 weeks, 6 phases, milestones M1–M6): conference paper, ends at Milestone M6 (EMBC submitted).
- **Journal track** (~12 months, 8 phases, milestones MJ1–MJ7): builds on the EMBC track, adds wPLI robustness, extended ablations, full interpretability, journal manuscript.

Milestones in the WBS (rows marked 🏁, e.g. M1–M6, MJ1–MJ7) are **supervisor sign-off gates**, distinct from and in addition to the per-stage stopping points above. Reaching the end of a numbered stage in this file is not the same as clearing a WBS milestone.

Three distinct states must be tracked separately and never conflated:

1. **Technical completion** — the code for a stage/task ran successfully and produced output. The agent may assess and report this on its own.
2. **Quality-gate completion** — the relevant Section 25 quality gate and/or WBS Quality Gates sheet items for that stage/task are satisfied. The agent may assess and report this against the documented criteria, but this is a claim about whether criteria are met, not an approval.
3. **Supervisor sign-off** — a human (per the WBS's "Owner" column, typically the supervisor) has explicitly approved the milestone. **The agent must never mark this complete on its own behalf, regardless of how strong 1 and 2 are.** It may only report "technically complete and quality-gate criteria appear satisfied — awaiting supervisor sign-off."

The agent must:

- Treat a WBS milestone row as blocking — do not begin the next WBS phase's tasks until that milestone's supervisor sign-off is recorded by a human, even if states 1 and 2 above are both satisfied.
- Never infer or assume sign-off from silence, from the human proceeding to a new topic, or from a prior conversation — sign-off must be explicit and stage-specific.

---

## 5. Raw Data Protection

The directory:

`data/raw/`

must be treated as READ-ONLY.

Never:

- Delete raw EEG files
- Rename raw EEG files
- Modify raw EEG files
- Overwrite raw EEG files
- Move raw EEG files
- Store generated outputs inside the raw dataset

All derived data must be written outside `data/raw/`.

Suggested locations:

`data/processed/`

`data/interim/`

`outputs/`

`reports/`

The raw dataset must always remain recoverable and unchanged.

---

## 6. Dataset Validation Requirements

Before preprocessing, inspect the complete EEGMMIDB dataset.

Validate:

- Expected subject count
- Subject IDs
- Available runs per subject
- Missing runs
- Duplicate files
- Zero-byte files
- Unreadable EDF files
- File naming consistency
- Sampling frequencies
- EEG channel counts
- Channel names
- Recording durations
- EDF annotations
- Event availability
- Dataset inconsistencies

Where practical, verify dataset integrity using provided metadata and checksums.

Do not assume all subjects or recordings are valid merely because directories exist.

Generate machine-readable and human-readable audit reports.

### 6a. Documented EEGMMIDB defects — audit explicitly; never auto-exclude or auto-modify

The following are independently documented in the dataset's own tooling (MOABB) and in published papers using EEGMMIDB — they are not an assumption to verify from scratch, but a known-issues checklist the audit must explicitly test for:

- **Subject 89**: reported incorrect/mislabeled event annotations.
- **Subjects 88, 92, 100**: reported sampling rate of 128 Hz instead of the dataset-standard 160 Hz, and task/rest timing of 5.125 s / 1.375 s instead of the standard 4 s / 4 s.

Required Stage 1 procedure for S088, S089, S092, S100 (now frozen as part of the audit record):

1. **Verify** each reported issue against both authoritative documentation (official PhysioNet/EEGMMIDB records, the citing literature) and this local copy of the raw data (actual sampling rate, actual event annotations, actual run timing) — do not accept either source alone.
2. **Document affected runs and consequences** per subject: which specific runs are affected, what the concrete downstream consequence would be if used unmodified (e.g. incompatible sampling rate breaking a shared pipeline, mislabeled events corrupting the MI target for that subject).
3. **Propose** — do not apply — pre-registered inclusion/exclusion/correction rules for each affected subject (e.g. "exclude from MI decoding but retain for resting-state-only analyses," "resample to match cohort and flag in all downstream outputs," "exclude entirely"). Present these as a proposal in the audit report.
4. **Stop for explicit human approval** of the proposed rules before any downstream stage (preprocessing onward) treats these four subjects as included, excluded, or corrected in any way. The agent must not decide this itself, and must not default to silent inclusion or silent exclusion in the absence of approval — the pipeline should treat these subjects as blocked pending the decision, not proceed with an assumed default.

This satisfies the WBS Quality Gate requirement that "exclusion criteria [be] pre-registered before analysis; excluded subjects logged with reason."

Do not assume other subjects are unaffected merely because they aren't on this list — this is a starting checklist, not a guarantee of completeness.

---

## 7. EEGMMIDB Run Mapping

Never guess EEGMMIDB run meanings.

Before assigning tasks or labels, verify the official PhysioNet EEGMMIDB documentation.

Explicitly document the meaning of runs R01–R14.

Clearly distinguish:

- Baseline eyes-open recordings
- Baseline eyes-closed recordings
- Motor execution
- Motor imagery
- Left/right fist tasks
- Both fists/both feet tasks

Verify annotation mappings such as T0, T1, and T2 for the relevant run type.

Do not assume T1 and T2 have identical semantic meanings across different task groups without verification.

The final run mapping must be stored in configuration or documentation and must be reproducible.

The WBS's working assumption is Runs 1–2 = baseline rest (EO/EC), Runs 4,6,8,10,12,14 = MI runs — this must be independently verified against the official documentation before being relied on, per the rule above, not accepted on the WBS's authority alone.

---

## 8. Resting-State Methodological Requirement

The research objective involves resting-state EEG.

Before using EEGMMIDB baseline recordings as resting-state input, explicitly evaluate and document:

- What EEGMMIDB provides
- Whether eyes-open baseline is appropriate
- Whether eyes-closed baseline is appropriate
- Whether EO and EC should be analysed separately
- Whether they may be combined
- Limitations of calling these recordings "resting-state EEG"

Never fabricate or imply the existence of a separate five-minute resting-state recording if the dataset does not provide one.

Any limitation caused by EEGMMIDB's recording design must be explicitly reported.

The WBS's Journal track (task J2.2) requires EO vs EC to be tested separately as a robustness analysis regardless of which is used as the primary resting-state input — this does not replace the methodological evaluation above, it is an additional required experiment.

---

## 9. Preprocessing Requirements and Frozen Stage 2 Modules

Stage 2 is complete, reviewed, validated, and frozen. Its filtering, bad-channel detection, ICA, epoching, and QC modules **must not be modified unless explicitly requested**. One explicit Stage 3 refinement corrected the filter implementation to the existing WBS-configured sequence (notch → band-pass → resample to 128 Hz); this is the canonical frozen behavior. The requirements below remain the scientific and reproducibility standards for interpreting and using their outputs; they do not authorize further changes to the frozen implementation.

Preprocessing must be scientifically justified and reproducible.

The preprocessing pipeline should evaluate and appropriately implement:

- EEG channel selection
- Channel type assignment
- Montage handling
- Band-pass filtering
- Power-line noise handling
- Re-referencing
- Resampling where justified
- Artifact detection
- ICA where justified
- Bad-channel handling
- Epoch extraction
- Epoch rejection

All preprocessing parameters must be configurable.

Avoid unexplained hardcoded thresholds.

Record:

- Input file
- Subject
- Run
- Original sampling frequency
- Output sampling frequency
- Channels retained
- Channels removed
- ICA components removed where applicable
- Epochs created
- Epochs rejected
- Rejection percentage
- Reason for rejection

A previous experimental pipeline experienced excessive epoch rejection under overly strict amplitude thresholds.

Therefore:

- Do not blindly reuse old rejection thresholds.
- Inspect EEG amplitude distributions.
- Justify rejection criteria.
- Report subject-level rejection rates.
- Flag abnormal rejection rates.
- Never silently discard large portions of data.

### 9a. WBS default parameters — adopted starting points, with explicit constraints on deviation

The WBS specifies concrete starting parameters. These are adopted as the documented default/starting configuration:

- Notch filter at 60 Hz (line noise for a US-recorded dataset), zero-phase FIR bandpass 1–40 Hz.
- Target resample rate of 128 Hz after anti-aliasing filtering, applied post-notch/bandpass.
- Extended Infomax ICA with ICLabel for automated component rejection; log ICs removed per subject; flag any subject losing >50% signal variance.
- Average reference; resting-state segmented into 2 s non-overlapping epochs.
- Epoch rejection at ±100 µV, with rejection flagged if a subject exceeds 30% rejected, and a minimum of 30 clean epochs required per subject.
- Connectivity sparsification tested at top 10%, 15%, and 20% of edges retained, with the best threshold selected empirically and documented (this is a pilot task in the WBS, not a fixed choice).

Constraints on these defaults, all mandatory:

- They remain subject to QC and scientific justification per the base "do not blindly reuse" rule in Section 9 — check them against this dataset's actual amplitude distributions.
- **Any deviation from these defaults must be explicitly documented, and must never be made merely because it improves a downstream result.** A parameter change is justified by preprocessing/data-quality evidence (amplitude distributions, rejection rates, ICA behavior), never by "this made the regression metric better."
- **Filtering order must be validated explicitly**, not assumed correct because it matches the WBS's stated sequence (notch → bandpass → resample) — confirm this order avoids aliasing and filter-edge artifacts before applying it at scale.
- **Validated implementation order:** apply the 60 Hz notch before the 1–40 Hz zero-phase FIR band-pass, then resample 160 Hz recordings to 128 Hz with MNE's anti-aliasing resampling. This matches the WBS preprocessing specification; recordings natively at 128 Hz must bypass resampling and remain subject to the separate validation rule below.
- **Sampling-rate-specific behavior must be validated separately for any recording natively at 128 Hz** — this includes subjects 88, 92, and 100 flagged in Section 6a. Resampling a 160 Hz recording down to 128 Hz is not the same operation as encountering a recording already at 128 Hz (no anti-aliasing resample needed, different filter behavior near Nyquist). Do not apply the same resample step unconditionally to both cases, and do not treat a native-128 Hz recording as "already conformant" without checking filter/epoch behavior at that rate. This validation is independent of, and does not substitute for, the Section 6a inclusion/exclusion decision for those subjects.

---

## 10. Subject Identity Preservation

Subject identity must be preserved through every stage.

Every derived artifact must be traceable to:

- Subject ID
- Original run
- Task
- Condition
- Processing configuration

Never rely solely on array row order to establish subject identity.

Where appropriate, store explicit subject IDs alongside matrices and targets.

Before training any subject-level prediction model, verify exact alignment between:

- Subject IDs
- Resting-state features
- Connectivity matrices
- Graphs
- Continuous MI targets

---

## 11. MI Decoding and Continuous Target Generation

The continuous regression target must originate from a defensible MI task-decoding pipeline.

The Stage 3 baseline decoder is:

CSP + LDA

No alternative decoder may replace CSP + LDA during Stage 3 unless explicitly approved.

The MI decoding pipeline must:

- Use correctly mapped MI runs
- Use correctly mapped task events
- Preserve subject identity
- For the current validation scope, use S001 only with stratified 5-fold cross-validation
- Use leakage-safe cross-validation
- Fit CSP only on training folds
- Fit scaling only on training folds where applicable
- Fit LDA only on training folds
- Evaluate on held-out data

The Stage 3 deliverable is one continuous balanced-accuracy score for S001, calculated from held-out predictions across the stratified 5-fold validation. Cohort-wide target generation is out of scope until explicitly requested.

Store at minimum:

- Subject ID
- Number of valid trials
- Number of trials per class
- Cross-validation scheme
- Fold-level scores
- Mean score
- Standard deviation
- Target metric

Do not define the continuous target using performance measured on training data.

---

## 12. Leakage Prevention

Data leakage is a critical failure condition.

Any operation that learns information from data must be fitted only on the appropriate training partition.

This includes, where applicable:

- Scaling
- Normalization
- Feature selection
- CSP
- PCA
- Imputation
- Hyperparameter optimization
- Graph threshold selection if data-driven
- Learned preprocessing transformations

For subject-level prediction, the held-out subject must never influence model fitting or data-dependent feature selection.

If nested cross-validation is used:

Outer loop:
→ unbiased subject-level evaluation

Inner loop:
→ hyperparameter optimization and model selection

Never optimize hyperparameters directly on outer test subjects.

The WBS requires the exact same LOSO fold indices to be reused across every model (classical baselines, GCN, GAT) — this is a fairness requirement, not just a leakage requirement: fold indices must be generated once, saved to file, and loaded (not regenerated) for every subsequent model.

---

## 13. Regression Baselines

The primary downstream task is continuous MI performance prediction.

Candidate classical models include:

- Mean predictor / DummyRegressor
- Ridge Regression
- Elastic Net where justified
- Support Vector Regression
- Random Forest Regressor
- XGBoost Regressor

The WBS's required baseline set (RF, XGBoost, SVM, Ridge) must all be included and reported with no cherry-picking, per its Quality Gates sheet. This file adds a DummyRegressor/mean-predictor baseline on top of that as a sanity floor.

AutoML is not a mandatory V2 stage or WBS milestone (Section 1a). If used at all, it is optional exploratory analysis only, must be configured for REGRESSION, and must never delay or replace any WBS-required experiment or milestone.

Do not accidentally use:

- XGBClassifier
- Binary AutoML presets
- Classification metrics

for the primary continuous prediction task.

Every advanced model must be compared against simple baselines.

---

## 14. Subject-Level Evaluation

For predicting subject-level MI performance from resting-state EEG, evaluation must be subject-independent.

LOSO is a preferred outer evaluation strategy for the 109-subject dataset where computationally feasible.

All competing models should use identical predefined outer folds where possible.

Store fold assignments explicitly.

Primary regression metrics should include:

- MAE
- RMSE
- R²

Also report, where scientifically appropriate:

- Pearson correlation
- Spearman correlation

Do not rely on R² alone.

Report fold-level and aggregate performance, per the WBS Quality Gates requirement. Report aggregate regression metrics with appropriate 95% confidence intervals, preferably obtained using subject-level bootstrap resampling of out-of-fold predictions where appropriate. Report standard deviation separately where informative. Do not report or label mean ± SD as a 95% confidence interval.

---

## 15. Hyperparameter Optimization

Hyperparameter optimization must not leak test-subject information.

Preferred design:

Outer LOSO
→ hold out one subject

Inner CV
→ tune using training subjects only

Then:
→ train using selected parameters on outer training subjects
→ evaluate on held-out subject

If full nested LOSO is computationally prohibitive, any alternative must be explicitly documented and justified, and the estimated runtime of the full nested design must be reported before it is downgraded — do not silently substitute a cheaper design.

Random seeds must be fixed and recorded.

Optimization studies should be reproducible.

---

## 16. Functional Connectivity

The project should investigate resting-state functional connectivity.

Candidate connectivity measures include:

- PLV
- wPLI
- Coherence

Connectivity must be calculated consistently across subjects.

Document:

- Frequency band
- Windowing
- Epoching
- Connectivity estimator
- Matrix dimensions
- Symmetry
- Diagonal handling
- Thresholding
- Normalization

Avoid connectivity methods known to be strongly affected by volume conduction without discussing the limitation.

wPLI should be considered as a robustness alternative to PLV/coherence — the WBS's Quality Gates sheet makes wPLI-alongside-PLV a *required* gate for both tracks ("volume conduction confound addressed"), not optional. The EMBC track's initial connectivity computation is scoped to alpha and beta bands specifically; the Journal track (task J1.2) extends this to a full PLV vs wPLI vs coherence comparison across bands.

---

## 17. Graph Construction

For graph-based models:

Nodes:
→ EEG electrodes (64, per the WBS's specified architecture)

Edges:
→ Functional connectivity, thresholded per the sparsification values in Section 9a

Node features:
→ Band power per electrode (5 bands: delta, theta, alpha, beta, gamma), per the WBS

Graph construction must be deterministic and reproducible.

Store:

- Subject ID
- Node feature matrix
- Edge representation
- Connectivity matrix
- Graph construction parameters
- Continuous MI target

Check graph integrity before training.

Do not construct graphs using information from the prediction target.

---

## 18. GCN and GAT

Graph models must perform REGRESSION.

Expected output:

One continuous predicted MI performance value per subject.

Candidate models:

- GCN Regressor — primary graph model for the core/EMBC pipeline (Section 1a, resolved).
- GAT Regressor — secondary comparison/ablation model, scoped to the extended Journal track only, consistent with the WBS. Not a co-primary model in the EMBC track's Table 1. May only be promoted to co-primary by a later explicit human decision.

Graph models must be compared against non-graph regression baselines using identical subject-level evaluation splits.

Avoid claiming superiority based only on training performance.

The WBS additionally requires a topology ablation (thresholded vs. fully-connected graph) to isolate whether graph structure itself contributes anything beyond the node features — this is a required gate, not optional.

---

## 19. Statistical Validation

Publication-oriented results require statistical validation.

Include, where appropriate:

- Permutation testing
- Bootstrap confidence intervals
- Correlation significance
- Model comparison statistics

Permutation procedures must reflect the actual prediction problem.

For subject-level regression, target permutation should break the subject-level relationship between resting EEG predictors and MI performance while preserving the evaluation protocol. The WBS specifies n=1000 permutations per model as a required gate.

Report:

- Number of permutations
- Null distribution
- Observed metric
- p-value
- Random seed

Model comparisons (e.g. GCN/GAT vs. best classical baseline) do not use McNemar's test, which applies to classification only. A paired statistical comparison based on appropriate paired out-of-fold predictions or errors will be selected and justified before statistical analysis, with an effect size reported alongside it. Candidate methods include paired permutation testing or Wilcoxon signed-rank testing where their assumptions are satisfied; per Section 2a and Section 1b, this is not hard-coded in advance.

---

## 20. Ablation and Robustness Studies

Where computationally feasible, investigate:

- Eyes open vs eyes closed
- EO + EC where scientifically justified
- PLV vs wPLI vs coherence
- Frequency bands
- Resting-state recording duration (WBS Journal track tests 1 min vs 2 min vs 3 min explicitly)
- Connectivity thresholds (10%/15%/20%, per Section 9a)
- Graph density
- Node feature choices
- Classical ML vs graph models
- Different MI target-generation configurations
- Demographic confounds (age/sex from PhysioNet metadata, tested for correlation with the MI target — WBS Journal task J2.6)

The old WBS requirement for arbitrary 60%, 70%, and 80% binary threshold sensitivity should not define the primary analysis — it remains a required Journal-track *supplementary* robustness check (task J2.4: confirming whether model rankings are stable across thresholds), consistent with Section 2's treatment of binary framing as secondary only.

For the continuous regression pipeline, prioritize robustness of:

- Continuous target generation
- Decoder configuration
- Cross-validation
- Connectivity representation
- Regression model

Any binary threshold analysis should remain supplementary.

---

## 21. Reproducibility

Every major experiment must record:

- Configuration
- Random seed
- Input files
- Subject IDs
- Fold assignments
- Model parameters
- Software versions
- Output paths
- Metrics

Prefer configuration files under:

`configs/`

Do not scatter unexplained constants throughout scripts.

Use deterministic behavior where practical.

The WBS requires the public code repository to reproduce the primary results table in under one hour from a fresh environment before submission — pinned dependency versions (Section 22a) are necessary for this, not optional polish.

---

## 22. Output Organization

Use clear separation between source data and generated outputs.

Recommended structure:

data/raw/
→ immutable original EEG

data/interim/
→ temporary structured data

data/processed/
→ validated preprocessing outputs

outputs/targets/
→ continuous MI performance targets

outputs/features/
→ spectral/tabular features

outputs/connectivity/
→ connectivity matrices

outputs/graphs/
→ graph datasets

outputs/models/
→ saved models

outputs/results/
→ experiment metrics and predictions

outputs/figures/
→ publication figures (≥300 dpi, legible at single-column width, per WBS Quality Gates)

reports/
→ audits and human-readable reports

### 22a. Environment and dependencies

Per the WBS (task 1.1), the environment is:

- Python 3.10+
- CUDA (for GPU-accelerated training)
- PyTorch
- PyTorch Geometric (PyG)
- MNE-Python (EDF reading, preprocessing, ICA)
- Braindecode

Pin all dependency versions in `requirements.txt` at the repository root. The WBS's own acceptance criterion for environment setup is `pip install` completing and `import torch_geometric` succeeding — verify this explicitly before graph-model work begins, don't assume it from a successful install log alone.

---

## 23. Code Quality

Code should be:

- Modular
- Readable
- Reproducible
- Configurable
- Testable

Avoid unnecessary monolithic scripts.

Prefer separate modules for:

- Dataset handling
- Preprocessing
- MI decoding
- Target generation
- Feature extraction
- Connectivity
- Regression
- Graph construction
- GNN models
- Evaluation
- Statistics

Add tests for critical data transformations and alignment logic.

---

## 24. Agent Safety Rules

Before modifying code:

1. Inspect the repository.
2. Understand existing files.
3. Read this AGENTS.md.
4. Inspect the WBS.
5. Identify the explicitly requested stage.
6. Check Section 1a for resolved decisions and Section 1b for open items relevant to that stage before proceeding.

Never delete existing research results without explicit permission.

Never modify raw data.

Never silently overwrite important outputs.

Never invent missing scientific information.

Never guess dataset event mappings.

Never report an experiment as successful solely because code executed without errors.

Distinguish clearly between:

- Code implemented
- Code tested
- Pipeline executed
- Quality gate passed
- Scientific result obtained

If a methodological decision is uncertain, document the uncertainty and stop for review when it could materially affect downstream validity. This explicitly includes both items in Section 1b — do not default to an assumption for either just because a stage is ready to start.

---

## 25. Research Quality Gates

The pipeline should not advance merely because code runs.

Each stage must satisfy its relevant quality gate.

Examples:

Dataset audit:
→ 109 expected subjects accounted for or deviations documented; Section 6a known-issue subjects (S088, S089, S092, S100) explicitly verified against documentation and raw data, affected runs/consequences documented, and inclusion/exclusion/correction rules proposed and awaiting human approval — not yet applied

Preprocessing:
→ output integrity verified and abnormal rejection rates investigated

MI decoding:
→ event mapping validated and leakage-safe evaluation confirmed

Continuous targets:
→ for the current Stage 3 validation scope, S001 has one traceable, reliable balanced-accuracy target; at cohort-wide target generation, all included subjects have traceable, reliable target values

Feature extraction:
→ no unexplained NaN/Inf and exact subject alignment

Connectivity:
→ matrices valid, consistent, and traceable; wPLI computed alongside PLV per WBS gate

Regression:
→ evaluated against simple baselines with leakage-safe subject splits, using identical LOSO fold indices across all models

GCN/GAT:
→ graph integrity verified and compared fairly with baselines; topology ablation (thresholded vs. fully-connected) completed

Statistics:
→ null hypothesis testing and uncertainty reported

Publication:
→ all claims traceable to reproducible experiments; full Quality Gates sheet (Sheet 4) reviewed with Section 2a's reconciliation applied before submission

---

## 26. Current Validated Project State

- **Stage 1 — Dataset audit and run-mapping validation:** complete and frozen.
- **Stage 2 — Preprocessing and preprocessing QC:** complete, reviewed, validated, and frozen. Filtering, bad-channel detection, ICA, epoching, and QC are frozen modules and must not be modified unless explicitly requested; the authorized WBS-conformance correction to filtering order and resampling is included in the frozen state.
- **Stage 3 — MI decoding target generation:** current stage. Its objective is to generate one continuous MI decoding score for S001 using CSP + LDA, stratified 5-fold cross-validation, and balanced accuracy.

Do not modify code from Stage 1 or Stage 2 while working on Stage 3 unless the request explicitly authorizes it. Do not proceed beyond Stage 3 unless instructed.
