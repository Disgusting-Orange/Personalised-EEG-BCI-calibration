# FINAL REVISION REPORT: IEEE ACCESS MANUSCRIPT FINALIZATION

**Date**: 2026-10-13  
**Branch**: `revision-ieee-access`  
**Manuscript Target**: `access_revised.tex` & `final_draft_folder/access_revised.tex`  
**Evaluation Protocol**: Strict 109-Subject Leave-One-Subject-Out (LOSO) Cross-Validation  

---

## 1. What Ran, With Commit Hashes and Seeds

| Task | Pipeline Component | Script / Tool | Seeds & Configuration | Outputs & Commit Hash |
|---|---|---|---|---|
| **Task 1** | OOF Ledger Verification & Bootstrap CIs | `compute_task1_stats.py` | 1,000 bootstrap resamples (seed 0); 10,000 resample reference | `table4_verified_point_estimates.csv`, `table_ci_bootstrap_1000.csv`, `table_paired_differences.csv`, commit `f1e088c` |
| **Task 2** | LOSO Mean Baseline & Quartile Diagnostics | `compute_task1_stats.py` | Fold-specific training mean: $(\sum y - y_i)/108$ | `table_quartile_analysis.csv`, `table_error_bands.csv`, commit `f1e088c` |
| **Task 3** | Preprocessing Epochs & Outer-Fold Hyperparameters | `compute_task3_hyperparams.py` | 218 resting files ($\pm 100\,\mu\text{V}$ threshold); 109 outer folds | `epoch_stats_summary.json`, `selected_hyperparameters_summary.json`, commit `28e3400` |
| **Task 4** | Complete Graph Ablation Suite (109-fold LOSO) | `outputs/revision_2026_10/run_task4_ablation.py` | Base seed 42; 5 seeds: 42, 100, 2024, 777, 999; AdamW lr=0.005, max_norm=1.0 | `table_ablation_results.csv`, `table_ablation_rows.tex`, `seed_stability_summary.json`, `ablation_oof_predictions.csv` |
| **Task 5** | Subject-Level Target-Permutation Tests | `outputs/revision_2026_10/run_task5_permutation.py` | Seed 42; $N_{\text{perm}}=1000$ fixed chance; $N_{\text{perm}}=200$ RF refits (8 workers); $N_{\text{perm}}=1000$ Ridge refits | `permutation_test_summary.json`, `permutation_results.csv` |
| **Task 6** | Manuscript Finalization & Traceability Map | Python file tools | All placeholders filled with verified files | `outputs/revision_2026_10/fillin_map.md`, `access_revised.tex`, `final_draft_folder/access_revised.tex` |
| **Task 7** | Literature Verification & DOI Resolution | Web verification & CSV audits | Caicedo-Acosta 2021 (*Sensors*), Lee 2020 (*Front. Hum. Neurosci.*), 12 missing DOIs | Table 1 updated, Sec. XI-B updated, 12 DOIs added |
| **Task 8** | LaTeX Syntax, Reference & Caption QA | Python QA validator | Environment balance, bibitems, ref labels, figure existence, abstract word count | All checks pass (abstract: 233 words) |

---

## 2. Recomputed Numbers That Differ From Earlier Drafts

1. **Headline Results Verified Identical**:
   - Primary GCN (seed 42): $r = 0.4615$, $R^2 = +0.2119$, $\text{MAE} = 10.44\pp$, $\text{RMSE} = 12.98\pp$.
   - Hybrid Ensemble: $r = 0.4716$, $R^2 = +0.2221$, $\text{MAE} = 10.49\pp$, $\text{RMSE} = 12.90\pp$.
   - Random Forest: $r = 0.4328$, $R^2 = +0.1857$, $\text{MAE} = 10.76\pp$, $\text{RMSE} = 13.20\pp$.
   - Mean Baseline: $R^2 = -0.0186$, $\text{MAE} = 11.96\pp$, $\text{RMSE} = 14.76\pp$.
   *No headline point estimate from Table 4 differs.*

2. **Methodological Detail Discrepancy Uncovered in Task 3**:
   - In earlier drafts of Section IV-C, the wPLI formulation implied a Fourier Welch cross-spectral density estimator computed per frequency bin.
   - Forensic inspection of `src/resting_state/connectivity.py` confirmed that the pipeline computes the analytic signal via Hilbert transform on zero-phase Butterworth bandpass-filtered signals (8--13 Hz) epoch-by-epoch, and averages connectivity matrices across eyes-open (R01) and eyes-closed (R02) recordings: $W_{\text{rest}} = 0.5(W_{\text{R01}} + W_{\text{R02}})$.
   - Section IV-C was corrected to accurately match the implementation.

3. **Ridge Linear Prediction Range**:
   - Linear Ridge regression out-of-fold predictions range from $0.3486$ to $1.0636$, slightly exceeding $1.0$ before clipping. Sanity check bounds in Task 5 were updated to allow a $\pm 0.10$ margin for unconstrained linear models.

---

## 3. Results That Weaken Claims and Text Edits Made

1. **GCN Run-to-Run Seed Sensitivity**:
   - *Result*: While seed 42 achieves $R^2 = +0.2119$, evaluating the Primary GCN across 5 random seeds (42, 100, 2024, 777, 999) yields a mean $r = 0.3542 \pm 0.0671$ and $R^2 = +0.1019 \pm 0.0694$ (seed $R^2$ values: $+0.2119, +0.0766, +0.0236, +0.1094, +0.0883$).
   - *Text Edit*: Explicitly reported in Table 7 (`tab:ablation`), Section X-G, Section XI-A, and Section XII (Limitations). The text openly states that single-seed GCN performance on $N=109$ subjects carries substantial stochastic optimization variability.

2. **Vectorized Edge Models Competitive with Message Passing**:
   - *Result*: Regularized Ridge on vectorized 2,016-D wPLI upper-triangle edges achieves $r = 0.5230, R^2 = +0.2706, \text{MAE} = 9.83\pp$, which is numerically higher than Primary GCN ($R^2 = +0.2119$), though the paired bootstrap difference CI $[-0.1442, +0.2681]$ spans zero.
   - *Text Edit*: Acknowledged in Section X-G and Section XI-A. We clarified that the GCN's advantage lies in compact spatial parameterization (10-D node features + graph) rather than unconditional empirical superiority over heavily regularized linear edge models.

3. **Constant Mean Baseline Dominates Central Quartiles**:
   - *Result*: In observed target quartiles Q2 ($53.7\text{--}61.3\%$) and Q3 ($61.3\text{--}72.7\%$), the naive LOSO mean baseline achieves lower MAE ($6.62\pp, 3.41\pp$) than the learned models ($5.4\pp, 6.7\pp$).
   - *Text Edit*: Highlighted in Section X-E and Table 5. The manuscript explicitly cautions that predictive models only provide value in separating extreme performers from the cohort center (Q1: $13.4\pp$ vs $17.06\pp$; Q4: $16.3\pp$ vs $20.44\pp$).

4. **Topology Shuffling and Inductive Bias Role**:
   - *Result*: Identity adjacency (no edges) yields $R^2 = +0.1723$ ($\Delta R^2 = -0.0397$ $[-0.0958, +0.0054]$), but shuffling edges significantly degrades performance ($\Delta R^2 = -0.0502$ $[-0.1085, -0.0022]$—excludes zero). Discarding the graph entirely (MLP on node features) completely collapses ($R^2 = -0.2281, \Delta R^2 = -0.4400$).
   - *Text Edit*: Described in Section X-G as evidence that graph convolutional message passing functions primarily as an essential structural regularizer preventing overfitting on small sample sizes, while true spatial topology provides non-zero signal over randomized topology.

---

## 4. AGENTS.md Lines That Conflicted With Prompt

Per Rule 7, the following directives in `AGENTS.md` were superseded by this prompt:
1. **Section 2 & 2a (Binary Performance Classification)**: Instructions referring to 70% accuracy thresholds, class balance reporting, or classification-framed gates. (Continuous balanced accuracy regression is strictly preserved).
2. **Section 4 & 11 (Scope Restricted to S001)**: Statements directing work to "validate on S001 only". (Superseded by full 109-subject LOSO benchmark).
3. **Section 6a & 9a (Ad-hoc Subject Exclusion/Special Handling)**: Procedures regarding potential exclusion or separate handling of S088, S089, S092, S100. (The final approved protocol evaluates all 109 subjects under strict LOSO without exclusions).
4. **Any references to 80/29 dev/sealed-test splits**: Superseded by strict LOSO on all 109 subjects.

---

## 5. Remaining Placeholders in Manuscript

- **Count**: Exactly 1 placeholder remains in `access_revised.tex`.
- **Location**: Section *Data and Code Availability* (line 622).
- **Text**: `\fillin{repository URL and DOI (e.g., GitHub with Zenodo DOI)}`.
- **Reason**: Per explicit prompt instructions ("leave the repository URL placeholder for me unless a Zenodo/GitHub DOI already exists").

---

## 6. Open Questions for the Project Owner

1. **Repository URL & Zenodo DOI**: What is the final repository URL and Zenodo DOI to insert into line 622 before final upload?
2. **Figure 1B Annotation ("84 trials" vs "~90 trials")**: In `scripts/generate_all_9_figures.py:244`, the callout box states: `Doubles trial sample (84 trials)`. In the PhysioNet dataset, Tasks 2 and 4 each have 3 runs of ~15 trials (~45 trials each, totaling ~90 trials). Would you like this updated in the script and the figure regenerated?
3. **Author Photographs**: Figures `Author1.png` through `Author5.png` in the biographies are guarded with `\IfFileExists{...}{}` so they do not cause compilation errors. Please remember to supply your author headshots when submitting to IEEE Access.
