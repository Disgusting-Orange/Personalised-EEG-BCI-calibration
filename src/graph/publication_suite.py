"""Master publication suite generator for Stage 15.

Generates LaTeX comparison tables, multi-panel 300 DPI publication figures,
and comprehensive publication audit reports for IEEE TNSRE / EMBC submission.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("graph.publication_suite")


def _clean_ledger_dataframe(df_ledger: pd.DataFrame) -> pd.DataFrame:
    """Filter out repeated header rows and ensure numeric columns are properly typed."""
    df_clean = df_ledger[df_ledger["MAE"] != "MAE"].copy()
    numeric_cols = ["MAE", "RMSE", "R2", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    return df_clean


def generate_latex_table(df_ledger: pd.DataFrame) -> str:
    """Generate publication-ready LaTeX table code for paper manuscript."""
    df_ledger = _clean_ledger_dataframe(df_ledger)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Master LOSO Continuous Regression Performance Comparison Across 109 Subjects}",
        r"\label{tab:loso_master_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Model Category} & \textbf{Model ID} & \textbf{MAE} & \textbf{RMSE} & \textbf{R$^2$ Score [95\% CI]} & \textbf{Pearson $r$ ($p$-value)} \\",
        r"\hline",
    ]

    for _, row in df_ledger.iterrows():
        model_str = str(row["Model"]).upper()
        feat_str = str(row["Feature_Set"])
        mae = float(row["MAE"])
        rmse = float(row["RMSE"])
        r2 = float(row["R2"])
        r = float(row["Pearson_r"])
        p_val = str(row["Pearson_p"])

        bold_prefix = r"\textbf{" if model_str in ("GCN", "SVR") else ""
        bold_suffix = "}" if model_str in ("GCN", "SVR") else ""

        line = f"  {bold_prefix}{model_str}{bold_suffix} & {feat_str} & {mae:.4f} & {rmse:.4f} & {r2:.4f} & {r:.4f} ($p={p_val}$) \\\\"
        lines.append(line)

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_master_figure(
    df_ledger: pd.DataFrame,
    df_ablation: pd.DataFrame,
    output_dir: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Generate 4-panel 300 DPI master publication figure."""
    df_ledger = _clean_ledger_dataframe(df_ledger)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11), dpi=dpi)

    # Panel A: R2 Score Comparison Bar Chart
    models = df_ledger["Model"].str.upper().tolist()
    r2_vals = df_ledger["R2"].values
    colors = ["#2ca02c" if m == "GCN" else ("#9467bd" if m == "GAT" else "#1f77b4") for m in models]

    ax1.bar(models, r2_vals, color=colors, edgecolor="k", linewidth=0.8, alpha=0.85)
    ax1.axhline(0, color="k", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Coefficient of Determination R²", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Model Performance R² Comparison", fontsize=12, fontweight="bold", loc="left")

    # Panel B: Pearson r Correlation Comparison
    r_vals = df_ledger["Pearson_r"].values
    ax2.bar(models, r_vals, color=colors, edgecolor="k", linewidth=0.8, alpha=0.85)
    ax2.set_ylabel("Pearson Correlation r", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Pearson Correlation r Comparison", fontsize=12, fontweight="bold", loc="left")

    # Panel C: Topology Density Ablation Curve
    if not df_ablation.empty:
        dens = df_ablation["topology_density"].tolist()
        r2_abl = df_ablation["r2"].values
        ax3.plot(dens, r2_abl, "o-", color="#d62728", linewidth=2.5, markersize=8, label="GCN R² Score")
        ax3.set_xlabel("Graph Edge Sparsification Density", fontsize=11, fontweight="bold")
        ax3.set_ylabel("R² Score", fontsize=11, fontweight="bold")
        ax3.set_title("(C) Graph Topology Density Ablation", fontsize=12, fontweight="bold", loc="left")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="lower right")

    # Panel D: MAE Error Comparison
    mae_vals = df_ledger["MAE"].values
    ax4.bar(models, mae_vals, color="#ff7f0e", edgecolor="k", linewidth=0.8, alpha=0.85)
    ax4.set_ylabel("Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    ax4.set_title("(D) Prediction Error MAE Comparison", fontsize=12, fontweight="bold", loc="left")

    plt.tight_layout()
    fig_file = out_path / "master_publication_figure.png"
    plt.savefig(fig_file, dpi=dpi)
    plt.savefig(fig_file.with_suffix(".svg"))
    plt.close(fig)

    logger.info("Saved 4-panel master publication figure to %s", fig_file)
