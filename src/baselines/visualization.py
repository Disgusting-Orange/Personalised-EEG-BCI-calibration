"""Publication-quality visualization module for Stage 8 Baseline Regressors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def plot_predicted_vs_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    model_id: str,
    metrics: dict[str, float],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Generate publication-quality Predicted vs Ground-Truth scatter plot."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

    # Scatter plot
    ax.scatter(y_true, y_pred, color="#1f77b4", alpha=0.75, edgecolors="k", linewidth=0.5, label="Subjects (OOF)")

    # Reference line y = x
    min_val = min(y_true.min(), y_pred.min()) - 0.02
    max_val = max(y_true.max(), y_pred.max()) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Fit (y = x)")

    # Linear trend line
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    ax.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], "b-", linewidth=1.2, alpha=0.7, label="Linear Fit")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Ground-Truth MI Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted MI Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"Stage 8: {model_id.upper()} — Predicted vs Actual (LOSO OOF)", fontsize=12, fontweight="bold", pad=12)

    # Metrics annotation box
    textstr = (
        f"MAE: {metrics.get('mae', 0):.4f}\n"
        f"RMSE: {metrics.get('rmse', 0):.4f}\n"
        f"R²: {metrics.get('r2', 0):.4f}\n"
        f"Pearson r: {metrics.get('pearson_r', 0):.4f} (p={metrics.get('pearson_p', 1):.3e})"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9, edgecolor="grey")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    model_id: str,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Generate residual error distribution histogram and Q-Q plot."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    # Residual Histogram + KDE
    sns.histplot(residuals, kde=True, ax=ax1, color="#2ca02c", bins=20, stat="density")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Residual Error (Actual - Predicted)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax1.set_title(f"{model_id.upper()} Residual Error Distribution", fontsize=11, fontweight="bold")

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f"{model_id.upper()} Residual Q-Q Plot", fontsize=11, fontweight="bold")

    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_feature_importance(
    df_importance: pd.DataFrame,
    model_id: str,
    top_k: int = 20,
    output_path: Union[str, Path] | None = None,
    dpi: int = 300,
) -> None:
    """Generate horizontal bar plot of top K features by importance weight."""
    if df_importance.empty:
        return

    top_df = df_importance.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.barh(top_df["feature"], top_df["importance"], color="#ff7f0e", edgecolor="k", linewidth=0.5)

    ax.set_xlabel("Feature Importance Weight", fontsize=11, fontweight="bold")
    ax.set_ylabel("Feature Name", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_id.upper()} — Top {top_k} Informative Features", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
        plt.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_model_comparison(
    summary_list: list[dict[str, Any]],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Generate publication comparison bar plot across baseline models with 95% CIs."""
    df_summary = pd.DataFrame(summary_list)
    if df_summary.empty:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10), dpi=dpi)
    models = df_summary["model_id"].str.upper().tolist()

    # 1. MAE Comparison
    maes = df_summary["mae"].values
    mae_errs = np.zeros((2, len(maes)))
    for i, row in df_summary.iterrows():
        ci = row.get("mae_ci", (row["mae"], row["mae"]))
        mae_errs[0, i] = max(0.0, row["mae"] - ci[0])
        mae_errs[1, i] = max(0.0, ci[1] - row["mae"])

    ax1.bar(models, maes, yerr=mae_errs, capsize=5, color="#1f77b4", edgecolor="k", alpha=0.85)
    ax1.set_ylabel("MAE (Lower is better)", fontsize=10, fontweight="bold")
    ax1.set_title("Mean Absolute Error (95% CI)", fontsize=11, fontweight="bold")

    # 2. RMSE Comparison
    rmses = df_summary["rmse"].values
    rmse_errs = np.zeros((2, len(rmses)))
    for i, row in df_summary.iterrows():
        ci = row.get("rmse_ci", (row["rmse"], row["rmse"]))
        rmse_errs[0, i] = max(0.0, row["rmse"] - ci[0])
        rmse_errs[1, i] = max(0.0, ci[1] - row["rmse"])

    ax2.bar(models, rmses, yerr=rmse_errs, capsize=5, color="#d62728", edgecolor="k", alpha=0.85)
    ax2.set_ylabel("RMSE (Lower is better)", fontsize=10, fontweight="bold")
    ax2.set_title("Root Mean Squared Error (95% CI)", fontsize=11, fontweight="bold")

    # 3. R2 Comparison
    r2s = df_summary["r2"].values
    r2_errs = np.zeros((2, len(r2s)))
    for i, row in df_summary.iterrows():
        ci = row.get("r2_ci", (row["r2"], row["r2"]))
        r2_errs[0, i] = max(0.0, row["r2"] - ci[0])
        r2_errs[1, i] = max(0.0, ci[1] - row["r2"])

    ax3.bar(models, r2s, yerr=r2_errs, capsize=5, color="#2ca02c", edgecolor="k", alpha=0.85)
    ax3.set_ylabel("R² Score (Higher is better)", fontsize=10, fontweight="bold")
    ax3.set_title("Coefficient of Determination R² (95% CI)", fontsize=11, fontweight="bold")

    # 4. Pearson r Comparison
    pearsons = df_summary["pearson_r"].values
    pearson_errs = np.zeros((2, len(pearsons)))
    for i, row in df_summary.iterrows():
        ci = row.get("pearson_ci", (row["pearson_r"], row["pearson_r"]))
        pearson_errs[0, i] = max(0.0, row["pearson_r"] - ci[0])
        pearson_errs[1, i] = max(0.0, ci[1] - row["pearson_r"])

    ax4.bar(models, pearsons, yerr=pearson_errs, capsize=5, color="#9467bd", edgecolor="k", alpha=0.85)
    ax4.set_ylabel("Pearson Correlation r (Higher is better)", fontsize=10, fontweight="bold")
    ax4.set_title("Pearson Correlation r (95% CI)", fontsize=11, fontweight="bold")

    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
