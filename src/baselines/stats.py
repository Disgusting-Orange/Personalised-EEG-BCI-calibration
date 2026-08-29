"""Statistical evaluation, bootstrap confidence intervals, and significance testing for Stage 8.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

logger = logging.getLogger("baselines.stats")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute comprehensive regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth continuous target vector.
    y_pred:
        Model predictions vector.

    Returns
    -------
    dict[str, float]:
        Dictionary of computed metric values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    exp_var = float(explained_variance_score(y_true, y_pred))

    # Pearson correlation
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        p_res = stats.pearsonr(y_true, y_pred)
        pearson_r, pearson_p = float(p_res.statistic), float(p_res.pvalue)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    # Spearman correlation
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        s_res = stats.spearmanr(y_true, y_pred)
        spearman_r, spearman_p = float(s_res.statistic), float(s_res.pvalue)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "median_ae": med_ae,
        "explained_variance": exp_var,
    }


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute non-parametric 95% bootstrap confidence intervals for primary metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target vector.
    y_pred:
        Model predictions vector.
    n_bootstraps:
        Number of bootstrap iterations (default 1000).
    ci_level:
        Confidence interval level (default 0.95).
    seed:
        Random seed.

    Returns
    -------
    dict[str, tuple[float, float]]:
        Dictionary mapping metric names to (lower_ci, upper_ci) bounds.
    """
    n_samples = len(y_true)
    rng = np.random.default_rng(seed)

    boot_mae: list[float] = []
    boot_rmse: list[float] = []
    boot_r2: list[float] = []
    boot_pearson: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_boot = y_true[idx]
        yp_boot = y_pred[idx]

        m = compute_regression_metrics(yt_boot, yp_boot)
        boot_mae.append(m["mae"])
        boot_rmse.append(m["rmse"])
        boot_r2.append(m["r2"])
        boot_pearson.append(m["pearson_r"])

    alpha_lower = ((1.0 - ci_level) / 2.0) * 100
    alpha_upper = (1.0 - ((1.0 - ci_level) / 2.0)) * 100

    cis = {
        "mae_ci": (float(np.percentile(boot_mae, alpha_lower)), float(np.percentile(boot_mae, alpha_upper))),
        "rmse_ci": (float(np.percentile(boot_rmse, alpha_lower)), float(np.percentile(boot_rmse, alpha_upper))),
        "r2_ci": (float(np.percentile(boot_r2, alpha_lower)), float(np.percentile(boot_r2, alpha_upper))),
        "pearson_ci": (float(np.percentile(boot_pearson, alpha_lower)), float(np.percentile(boot_pearson, alpha_upper))),
    }
    return cis


def paired_model_comparison(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """Perform paired statistical comparison of prediction errors between two models.

    Parameters
    ----------
    y_true:
        Ground-truth target vector.
    y_pred_a, y_pred_b:
        Predictions vectors from Model A and Model B.

    Returns
    -------
    dict[str, float]:
        Wilcoxon signed-rank test statistic and p-value.
    """
    err_a = np.abs(y_true - y_pred_a)
    err_b = np.abs(y_true - y_pred_b)

    diff = err_a - err_b
    if np.allclose(diff, 0.0, atol=1e-8):
        return {"stat": 0.0, "p_value": 1.0, "mean_diff": 0.0}

    res = stats.wilcoxon(err_a, err_b)
    return {
        "stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff": float(np.mean(diff)),
    }
