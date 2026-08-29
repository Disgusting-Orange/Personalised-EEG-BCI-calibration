"""Publication graphics generator for Stage 14 Statistical Analysis.

Renders 300 DPI Bland-Altman agreement plots, residual error distribution & Q-Q plots,
and predicted vs actual scatter plots with regression trendlines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

logger = logging.getLogger("graph.statistical_viz")


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Render 300 DPI Bland-Altman agreement plot between predictions and ground-truth targets."""
    means = (y_true + y_pred) / 2.0
    diffs = y_pred - y_true
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    ax.scatter(means, diffs, color="#1f77b4", alpha=0.7, edgecolors="k", linewidth=0.5, s=50)

    ax.axhline(mean_diff, color="red", linestyle="--", linewidth=1.5, label=f"Mean Bias ({mean_diff:+.4f})")
    ax.axhline(upper_loa, color="grey", linestyle=":", linewidth=1.5, label=f"+1.96 SD ({upper_loa:+.4f})")
    ax.axhline(lower_loa, color="grey", linestyle=":", linewidth=1.5, label=f"-1.96 SD ({lower_loa:+.4f})")

    ax.set_xlabel("Mean of Predicted and Actual Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Difference (Predicted - Actual)", fontsize=11, fontweight="bold")
    ax.set_title(f"Stage 14: Bland–Altman Plot ({model_name.upper()})", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Render 2-panel 300 DPI residual distribution histogram and Q-Q plot."""
    residuals = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    # Panel 1: Residual Histogram & KDE
    sns.histplot(residuals, kde=True, ax=ax1, color="#2ca02c", edgecolor="k", linewidth=0.8, stat="density")
    ax1.set_xlabel("Residual Error (Predicted - Actual)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Residual Distribution", fontsize=12, fontweight="bold", loc="left")
    ax1.axvline(0, color="k", linestyle="--", linewidth=1.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Residual Normal Q-Q Plot
    (osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist="norm")
    ax2.plot(osm, osr, "o", color="#9467bd", alpha=0.7, markeredgecolor="k", markeredgewidth=0.5, markersize=6)
    ax2.plot(osm, slope * osm + intercept, "r--", linewidth=1.5, label=f"Fit (r²={r_val**2:.4f})")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Ordered Residual Quantiles", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Normal Q-Q Plot", fontsize=12, fontweight="bold", loc="left")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)


def plot_predicted_vs_actual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    metrics: dict[str, Any],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Render 300 DPI publication scatter plot with linear trendline."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

    ax.scatter(y_true, y_pred, color="#1f77b4", alpha=0.7, edgecolors="k", linewidth=0.5, s=55, label="Held-Out Subjects")

    min_val = min(y_true.min(), y_pred.min()) - 0.02
    max_val = max(y_true.max(), y_pred.max()) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.2, label="Ideal (y = x)")

    # Linear trendline
    slope, intercept, r_val, p_val, _ = stats.linregress(y_true, y_pred)
    ax.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], "r-", linewidth=1.8, label=f"Trendline (r={r_val:.4f})")

    ax.set_xlabel("Actual Continuous Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Continuous Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"Stage 14: {model_name.upper()} Scatter Plot (N=109)", fontsize=12, fontweight="bold", pad=12)

    r2_val = metrics.get("r2", 0.0)
    mae_val = metrics.get("mae", 0.0)
    textstr = f"R² Score: {r2_val:.4f}\nPearson r: {r_val:.4f} (p={p_val:.4e})\nMAE: {mae_val:.4f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9, edgecolor="grey")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)
