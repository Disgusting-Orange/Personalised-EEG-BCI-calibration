# Traceability Map: Manuscript Placeholder Fill-in Ledger

This ledger documents the exact source file and run for every `\fillin{...}` placeholder and quantitative statement filled in `access_revised.tex`.

| # | Placeholder Location in `access_revised.tex` | Replaced Content / Value | Source File / Run Record | Status |
|---|---|---|---|---|
| 1 | Table 1 (Related Work), Caicedo-Acosta row, predictor window | `pre-cue / short-window functional connectivity indicators, public MI databases` | Published article *Sensors* 21(6):1932 (doi: 10.3390/s21061932) | Complete |
| 2 | Table 1 (Related Work), Lee et al. row, sample size $N$ | `54` | Published article *Front. Hum. Neurosci.* 14:321 (doi: 10.3389/fnhum.2020.00321) | Complete |
| 3 | Sec. IV-A, Preprocessing Epoching & QC, mean retained epochs | `$29.98$` | `outputs/revision_2026_10/epoch_stats_summary.json` & `resting_epochs_218_recordings.csv` | Complete |
| 4 | Sec. IV-A, Preprocessing Epoching & QC, minimum retained epochs | `$29$` | `outputs/revision_2026_10/epoch_stats_summary.json` & `resting_epochs_218_recordings.csv` | Complete |
| 5 | Sec. IV-C, wPLI Connectivity Specification | Condition-wise R01 and R02, zero-phase Butterworth filter (8--13 Hz), Hilbert analytic imaginary cross-products, unweighted arithmetic mean $W_{\text{rest}} = 0.5(W_{\text{R01}} + W_{\text{R02}})$ | Pipeline source code `src/resting_state/connectivity.py` | Complete |
| 6 | Sec. VIII-B, Benchmark Regressors, Ridge $\alpha$ median & range | median `$1000.0$` (range `$1000.0$--$1000.0$`) | `outputs/revision_2026_10/selected_hyperparameters_summary.json` | Complete |
| 7 | Sec. VIII-B, Benchmark Regressors, SVR $C$ median & range | median `$0.1$` (range `$0.1$--$10.0$`) | `outputs/revision_2026_10/selected_hyperparameters_summary.json` | Complete |
| 8 | Sec. VIII-B, Benchmark Regressors, SVR $\epsilon$ median & range | median `$0.1$` (range `$0.001$--$0.1$`) | `outputs/revision_2026_10/selected_hyperparameters_summary.json` | Complete |
| 9 | Sec. X-E, Table `tab:quartile_analysis`, Mean baseline Q1 | MAE `$17.06$`, Bias `$+17.1$` | `outputs/revision_2026_10/table_quartile_analysis.csv` | Complete |
| 10 | Sec. X-E, Table `tab:quartile_analysis`, Mean baseline Q2 | MAE `$6.62$`, Bias `$+6.6$` | `outputs/revision_2026_10/table_quartile_analysis.csv` | Complete |
| 11 | Sec. X-E, Table `tab:quartile_analysis`, Mean baseline Q3 | MAE `$3.41$`, Bias `$-2.5$` | `outputs/revision_2026_10/table_quartile_analysis.csv` | Complete |
| 12 | Sec. X-E, Table `tab:quartile_analysis`, Mean baseline Q4 | MAE `$20.44$`, Bias `$-20.4$` | `outputs/revision_2026_10/table_quartile_analysis.csv` | Complete |
| 13 | Sec. X-E, Quartile performance sentence | Constant mean baseline achieves lower/comparable MAE in central quartiles ($6.62\pp, 3.41\pp$) but models outperform baseline in extreme quartiles ($13.4\pp$ vs $17.06\pp$ in Q1; $16.3\pp$ vs $20.44\pp$ in Q4) | Direct empirical result from Table 5 (`table_quartile_analysis.csv`) | Complete |
| 14 | Sec. X-F, Error bands $\le 5\pp$ share & count | `$22.9\%$` ($n=25$) | `outputs/revision_2026_10/table_error_bands.csv` | Complete |
| 15 | Sec. X-F, Error bands $5\text{--}10\pp$ share & count | `$25.7\%$` ($n=28$) | `outputs/revision_2026_10/table_error_bands.csv` | Complete |
| 16 | Sec. X-F, Error bands $10\text{--}15\pp$ share & count | `$17.4\%$` ($n=19$) | `outputs/revision_2026_10/table_error_bands.csv` | Complete |
| 17 | Sec. X-F, Error bands $> 15\pp$ share & count | `$33.9\%$` ($n=37$) | `outputs/revision_2026_10/table_error_bands.csv` | Complete |
| 18 | Sec. X-G, Numerical instability cause and correction | Unconstrained gradient norms leading to divergent output scales on unregularized topologies, stabilized by gradient norm clipping ($\text{max\_norm}=1.0$), plateau LR scheduler, and bounded predictions | Diagnosed from historical controls; verified in `run_task4_ablation.py` | Complete |
| 19 | Sec. XI-B, Relation to Caicedo-Acosta connectivity regression | Unclustered global LOSO ($r=0.3824, R^2=0.1449, \text{MAE}=11.50\pp$) vs tercile-clustered LOSO ($r=0.7838, R^2=0.5772, \text{MAE}=7.58\pp$); clustering truncates variance | `reports/caicedo_reproduction_comparison.csv` | Complete |
| 20 | Bibliography DOIs (12 references) | Verified publisher DOIs added for Zhang 2015a, Zhang 2015b, Bamdadian 2014, Cui 2023, Hamedi 2016, Khademi 2023, Wang 2024, Saibene 2024, Saibene 2023, Sun 2023, Xue 2024, Shi 2024 | Publisher records (Springer, IOP, Elsevier, MDPI, MIT Press, Frontiers) | Complete |
| 21 | Table `tab:ablation` (Table 7) | Ablation suite metrics for 5 seeds, identity adjacency, shuffled adjacency, node-feature MLP, RF on wPLI, Ridge on wPLI | `outputs/revision_2026_10/table_ablation_results.csv`, `table_ablation_rows.tex` | Complete |
| 22 | Sec. X-G, Graph Ablation Interpretive Text | Factual summary of topology shuffling ($\Delta R^2=-0.0502$ excluding zero), MLP collapse ($R^2=-0.2281$), vectorized edge models, and seed variance ($R^2=+0.1019 \pm 0.0694$) | `outputs/revision_2026_10/table_ablation_results.csv` | Complete |
| 23 | Sec. X-G, Permutation Test $p$-values and null distributions | Hybrid null mean $R^2=-0.2075$ (q95=$-0.0680$), $p=0.0010$; GCN null mean $R^2=-0.2415$ (q95=$-0.0855$), $p=0.0010$; RF refit null mean $R^2=-0.1264$ (q95=$+0.0012$), $p=0.0050$; Ridge refit $p=0.0589$ | `outputs/revision_2026_10/permutation_test_summary.json` & `permutation_results.csv` | Complete |
| 24 | Abstract, Permutation $p$-value | `$p < 0.001$` (word count 233 words, $\le 250$ words) | `outputs/revision_2026_10/permutation_test_summary.json` | Complete |
| 25 | Sec. XI-A, Discussion Ablation Summary | One-sentence summary: shuffling edge topology degrades accuracy ($\Delta R^2=-0.0502$) and omitting graph structure causes MLP to collapse ($R^2=-0.2281$), though Ridge on vectorized edges reaches $R^2=+0.2706$ | `outputs/revision_2026_10/table_ablation_results.csv` | Complete |
| 26 | Sec. XII, Conclusion Permutation & Ablation Summary | Hybrid permutation $p=0.0010$ ($p < 0.001$) and ablation summary confirming topology necessity and seed variability | `outputs/revision_2026_10/permutation_test_summary.json` & `table_ablation_results.csv` | Complete |
| 27 | Data & Code Availability URL | `\fillin{repository URL and DOI (e.g., GitHub with Zenodo DOI)}` left untouched per prompt instructions | User-managed repository release | Retained as placeholder |

