"""Publication-quality statistical testing & validation framework for Stage 14.

Performs paired Wilcoxon signed-rank tests, paired t-tests, Holm-Bonferroni FDR correction,
Cohen's dz & rank-biserial effect sizes, 95% bootstrap confidence intervals,
and target permutation testing (N=1000).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("graph.statistical_tests")


def compute_cohens_dz(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Compute Cohen's dz effect size for paired samples: mean(diff) / std(diff)."""
    diff = err_a - err_b
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-9:
        return 0.0
    return float(np.mean(diff) / std_diff)


def compute_rank_biserial(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Compute Matched-Pairs Rank-Biserial Correlation (r_rb) for Wilcoxon test."""
    diff = err_a - err_b
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero_diff))
    pos_rank_sum = float(np.sum(ranks[nonzero_diff > 0]))
    neg_rank_sum = float(np.sum(ranks[nonzero_diff < 0]))
    total_rank_sum = pos_rank_sum + neg_rank_sum

    if total_rank_sum < 1e-9:
        return 0.0
    return float((pos_rank_sum - neg_rank_sum) / total_rank_sum)


def paired_model_comparisons(
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    ref_model: str = "gcn",
) -> pd.DataFrame:
    """Perform paired statistical significance tests comparing ref_model against all other models.

    Parameters
    ----------
    y_true:
        Ground-truth target vector (N=109).
    predictions_dict:
        Dictionary mapping model_id -> predicted vector (N=109).
    ref_model:
        Reference model ID (default 'gcn').

    Returns
    -------
    pd.DataFrame:
        Table of paired test statistics, p-values, Holm-Bonferroni adjusted p-values, and effect sizes.
    """
    if ref_model not in predictions_dict:
        raise KeyError(f"Reference model '{ref_model}' not found in predictions dictionary.")

    y_ref = predictions_dict[ref_model]
    err_ref = np.abs(y_true - y_ref)

    rows = []
    p_values = []

    model_keys = [k for k in predictions_dict.keys() if k != ref_model]

    for model_id in model_keys:
        y_other = predictions_dict[model_id]
        err_other = np.abs(y_true - y_other)

        diff = err_other - err_ref  # Positive if ref_model has lower absolute error

        # 1. Wilcoxon Signed-Rank Test (Non-parametric)
        w_stat, p_wilcoxon = stats.wilcoxon(err_ref, err_other)

        # 2. Paired t-Test (Parametric)
        t_stat, p_ttest = stats.ttest_rel(err_ref, err_other)

        # 3. Effect Sizes
        dz = compute_cohens_dz(err_other, err_ref)
        r_rb = compute_rank_biserial(err_other, err_ref)

        rows.append({
            "comparison": f"{ref_model.upper()} vs {model_id.upper()}",
            "ref_model": ref_model,
            "target_model": model_id,
            "mean_mae_diff": float(np.mean(diff)),
            "wilcoxon_stat": float(w_stat),
            "p_wilcoxon": float(p_wilcoxon),
            "ttest_stat": float(t_stat),
            "p_ttest": float(p_ttest),
            "cohens_dz": dz,
            "rank_biserial_r": r_rb,
        })
        p_values.append(p_wilcoxon)

    # Apply Holm-Bonferroni multiple comparison correction
    if p_values:
        sorted_indices = np.argsort(p_values)
        adj_p_values = np.zeros(len(p_values))
        m = len(p_values)
        for rank, idx in enumerate(sorted_indices):
            adj_p_values[idx] = min(1.0, p_values[idx] * (m - rank))

        for idx, adj_p in enumerate(adj_p_values):
            rows[idx]["p_adj_holm"] = float(adj_p)
            rows[idx]["significant_05"] = bool(adj_p < 0.05)

    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute non-parametric 95% bootstrap confidence intervals for all regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target array.
    y_pred:
        Model prediction array.
    n_bootstrap:
        Number of bootstrap resamples (default 1000).
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]:
        Point estimates and 95% CIs [2.5%, 97.5%] for MAE, RMSE, R², Pearson r, Spearman ρ.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    boot_mae = np.zeros(n_bootstrap)
    boot_rmse = np.zeros(n_bootstrap)
    boot_r2 = np.zeros(n_bootstrap)
    boot_pearson = np.zeros(n_bootstrap)
    boot_spearman = np.zeros(n_bootstrap)

    for k in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_b, yp_b = y_true[idx], y_pred[idx]

        boot_mae[k] = mean_absolute_error(yt_b, yp_b)
        boot_rmse[k] = np.sqrt(mean_squared_error(yt_b, yp_b))
        boot_r2[k] = r2_score(yt_b, yp_b)

        if np.std(yt_b) > 1e-9 and np.std(yp_b) > 1e-9:
            boot_pearson[k] = stats.pearsonr(yt_b, yp_b).statistic
            boot_spearman[k] = stats.spearmanr(yt_b, yp_b).statistic
        else:
            boot_pearson[k] = 0.0
            boot_spearman[k] = 0.0

    def _ci(arr: np.ndarray) -> list[float]:
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_ci": _ci(boot_mae),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmse_ci": _ci(boot_rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_ci": _ci(boot_r2),
        "pearson_r": float(stats.pearsonr(y_true, y_pred).statistic),
        "pearson_r_ci": _ci(boot_pearson),
        "spearman_r": float(stats.spearmanr(y_true, y_pred).statistic),
        "spearman_r_ci": _ci(boot_spearman),
    }


def target_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform target permutation test (N=1000) breaking subject-level association.

    Parameters
    ----------
    y_true:
        Ground-truth target array.
    y_pred:
        Model prediction array.
    n_permutations:
        Number of permutations (default 1000).
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]:
        Null distribution statistics and empirical p-values for R² and Pearson r.
    """
    rng = np.random.default_rng(seed)

    obs_r2 = float(r2_score(y_true, y_pred))
    obs_r = float(stats.pearsonr(y_true, y_pred).statistic)

    null_r2 = np.zeros(n_permutations)
    null_r = np.zeros(n_permutations)

    for k in range(n_permutations):
        y_perm = rng.permutation(y_true)
        null_r2[k] = r2_score(y_perm, y_pred)
        if np.std(y_perm) > 1e-9 and np.std(y_pred) > 1e-9:
            null_r[k] = stats.pearsonr(y_perm, y_pred).statistic

    p_r2 = float((1.0 + np.sum(null_r2 >= obs_r2)) / (n_permutations + 1.0))
    p_r = float((1.0 + np.sum(null_r >= obs_r)) / (n_permutations + 1.0))

    return {
        "n_permutations": n_permutations,
        "obs_r2": obs_r2,
        "obs_pearson_r": obs_r,
        "perm_r2_pvalue": p_r2,
        "perm_pearson_pvalue": p_r,
        "null_r2_mean": float(np.mean(null_r2)),
        "null_r2_std": float(np.std(null_r2)),
        "null_r_mean": float(np.mean(null_r)),
        "null_r_std": float(np.std(null_r)),
    }


def compute_loso_fold_robustness(
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute per-subject fold error statistics, variance, and standard deviation across models."""
    rows = []
    for model_id, y_pred in predictions_dict.items():
        errors = np.abs(y_true - y_pred)
        rows.append({
            "model_id": model_id.upper(),
            "mean_abs_error": float(np.mean(errors)),
            "std_abs_error": float(np.std(errors, ddof=1)),
            "variance_abs_error": float(np.var(errors, ddof=1)),
            "median_abs_error": float(np.median(errors)),
            "iqr_abs_error": float(np.percentile(errors, 75) - np.percentile(errors, 25)),
            "max_error": float(np.max(errors)),
        })
    return pd.DataFrame(rows).sort_values(by="mean_abs_error")
