"""Publication-Quality Figure & LaTeX Table Generator for Stage 16 Scientific Ablations.

Generates 300 DPI multi-panel composite figures, bar charts, sensitivity plots,
and publication-ready LaTeX tables detailing the 7 scientific ablation experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("graph.ablation_viz")


def generate_ablation_latex_table(df_results: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """Generate publication-ready LaTeX table for Stage 16 Ablations."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Scientific Ablation Studies of the Primary GCN Regressor across 109 Subjects under Leave-One-Subject-Out (LOSO) Cross-Validation. Bold entries demarcate the frozen primary baseline configuration.}",
        r"\label{tab:gcn_ablations}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Ablation Dimension} & \textbf{Variant / Component} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$ Score} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$} & \textbf{$\Delta R^2$ vs Base} \\",
        r"\midrule",
    ]

    base_r2_rows = df_results.loc[(df_results["Experiment"] == "topology_density") & (df_results["Variant"] == "top20"), "R2"].values
    base_r2 = float(base_r2_rows[0]) if len(base_r2_rows) > 0 else float(df_results["R2"].iloc[0])

    current_exp = ""
    for _, row in df_results.iterrows():
        exp_name = row["Experiment"].replace("_", " ").title()
        variant = str(row["Variant"])
        mae = float(row["MAE"])
        rmse = float(row["RMSE"])
        r2 = float(row["R2"])
        r_p = float(row["Pearson_r"])
        r_s = float(row["Spearman_r"])
        delta_r2 = r2 - base_r2

        # Format baseline in bold
        is_baseline = variant in ["top20", "wPLI", "Alpha", "Concat", "SmoothL1", "With_JK", "Cosine"]
        if is_baseline:
            variant_str = f"\\textbf{{{variant} (Primary)}}"
            r2_str = f"\\textbf{{{r2:.4f}}}"
            mae_str = f"\\textbf{{{mae:.4f}}}"
            rp_str = f"\\textbf{{{r_p:.4f}}}"
        else:
            variant_str = variant
            r2_str = f"{r2:.4f}"
            mae_str = f"{mae:.4f}"
            rp_str = f"{r_p:.4f}"

        if exp_name != current_exp:
            if current_exp != "":
                latex.append(r"\midrule")
            current_exp = exp_name
            exp_col = f"\\multirow{{1}}{{*}}{{\\textbf{{{exp_name}}}}}"
        else:
            exp_col = ""

        delta_str = f"{delta_r2:+.4f}" if not is_baseline else "Ref"
        latex.append(f"  {exp_col} & {variant_str} & {mae_str} & {rmse:.4f} & {r2_str} & {rp_str} & {r_s:.4f} & {delta_str} \\\\")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_str = "\n".join(latex)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    logger.info(f"Saved LaTeX ablation table to {out_file}")
    return latex_str


def generate_publication_ablation_figure(
    df_results: pd.DataFrame,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Generate 300 DPI 6-panel scientific ablation summary figure."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=dpi)

    palette = sns.color_palette("deep")
    primary_color = "#1f77b4"
    accent_color = "#2ca02c"
    highlight_color = "#d62728"

    # Panel A: Topology Density Sensitivity (Line Plot)
    ax_a = axes[0, 0]
    df_topo = df_results[df_results["Experiment"] == "topology_density"].copy()
    densities = [int(v.replace("top", "")) for v in df_topo["Variant"]]
    r2_vals = df_topo["R2"].values
    ax_a.plot(densities, r2_vals, marker="o", linewidth=2.5, markersize=8, color=primary_color, label="$R^2$ Score")
    if 20 in densities:
        ax_a.axvline(20, linestyle="--", color=highlight_color, alpha=0.7, label="Primary Top20")
    ax_a.set_title("(A) Graph Topology Density Sensitivity", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_xlabel("Edge Sparsification Density (%)", fontsize=10)
    ax_a.set_ylabel("Held-Out $R^2$ Score", fontsize=10)
    ax_a.set_xticks(densities)
    ax_a.legend(loc="lower right", frameon=True)
    ax_a.set_ylim(-0.05, 0.32)

    # Panel B: Connectivity Metric (Bar Plot)
    ax_b = axes[0, 1]
    df_conn = df_results[df_results["Experiment"] == "connectivity_metric"].copy()
    sns.barplot(data=df_conn, x="Variant", y="R2", ax=ax_b, palette=[primary_color, "#ff7f0e"])
    ax_b.set_title("(B) Functional Connectivity Metric", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_xlabel("Connectivity Index", fontsize=10)
    ax_b.set_ylabel("Held-Out $R^2$ Score", fontsize=10)
    ax_b.set_ylim(0, 0.32)
    for p in ax_b.patches:
        ax_b.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='bold')

    # Panel C: Spectral Frequency Band Contribution (Bar Plot)
    ax_c = axes[0, 2]
    df_band = df_results[df_results["Experiment"] == "frequency_band"].copy()
    band_colors = [palette[i % len(palette)] for i in range(len(df_band))]
    sns.barplot(data=df_band, x="Variant", y="R2", ax=ax_c, palette=band_colors)
    ax_c.set_title("(C) Spectral Frequency Band Contribution", fontsize=12, fontweight="bold", pad=10)
    ax_c.set_xlabel("EEG Frequency Band", fontsize=10)
    ax_c.set_ylabel("Held-Out $R^2$ Score", fontsize=10)
    ax_c.set_ylim(-0.05, 0.32)
    for p in ax_c.patches:
        val = p.get_height()
        y_pos = max(val, 0)
        ax_c.annotate(f"{val:.4f}", (p.get_x() + p.get_width() / 2., y_pos),
                      ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='bold')

    # Panel D: Readout Pooling Strategy (Bar Plot)
    ax_d = axes[1, 0]
    df_pool = df_results[df_results["Experiment"] == "pooling_strategy"].copy()
    sns.barplot(data=df_pool, x="Variant", y="R2", ax=ax_d, palette=[palette[0], palette[1], accent_color])
    ax_d.set_title("(D) Readout Pooling Strategy", fontsize=12, fontweight="bold", pad=10)
    ax_d.set_xlabel("Graph Readout Pooling", fontsize=10)
    ax_d.set_ylabel("Held-Out $R^2$ Score", fontsize=10)
    ax_d.set_ylim(0, 0.32)
    for p in ax_d.patches:
        ax_d.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='bold')

    # Panel E: Loss Function & JK Layer Aggregation (Bar Plot)
    ax_e = axes[1, 1]
    df_combo = pd.concat([
        df_results[df_results["Experiment"] == "loss_function"],
        df_results[df_results["Experiment"] == "jumping_knowledge"]
    ])
    sns.barplot(data=df_combo, x="Variant", y="R2", ax=ax_e, palette="crest")
    ax_e.set_title("(E) Loss Criterion & JK Layer Aggregation", fontsize=12, fontweight="bold", pad=10)
    ax_e.set_xlabel("Architectural / Objective Variant", fontsize=10)
    ax_e.set_ylabel("Held-Out $R^2$ Score", fontsize=10)
    ax_e.set_ylim(0, 0.32)
    for p in ax_e.patches:
        ax_e.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='bold')

    # Panel F: Relative Component Contribution Ranking (Horizontal Bar Chart)
    ax_f = axes[1, 2]
    base_r2_rows = df_results.loc[(df_results["Experiment"] == "topology_density") & (df_results["Variant"] == "top20"), "R2"].values
    base_r2 = float(base_r2_rows[0]) if len(base_r2_rows) > 0 else float(df_results["R2"].iloc[0])
    contributions = [
        ("Alpha Band (vs Delta)", base_r2 - float(df_results[(df_results["Experiment"]=="frequency_band") & (df_results["Variant"]=="Delta")]["R2"].iloc[0])),
        ("wPLI Connectivity (vs PLV)", base_r2 - float(df_results[(df_results["Experiment"]=="connectivity_metric") & (df_results["Variant"]=="PLV")]["R2"].iloc[0])),
        ("Dual Pool (vs Mean)", base_r2 - float(df_results[(df_results["Experiment"]=="pooling_strategy") & (df_results["Variant"]=="Mean")]["R2"].iloc[0])),
        ("JumpingKnowledge (vs No JK)", base_r2 - float(df_results[(df_results["Experiment"]=="jumping_knowledge") & (df_results["Variant"]=="Without_JK")]["R2"].iloc[0])),
        ("Cosine Scheduler (vs Plateau)", base_r2 - float(df_results[(df_results["Experiment"]=="learning_rate_scheduler") & (df_results["Variant"]=="Plateau")]["R2"].iloc[0])),
        ("SmoothL1 Loss (vs MSE)", base_r2 - float(df_results[(df_results["Experiment"]=="loss_function") & (df_results["Variant"]=="MSE")]["R2"].iloc[0])),
    ]
    labels_f = [c[0] for c in contributions]
    values_f = [c[1] for c in contributions]

    y_pos_f = np.arange(len(labels_f))
    ax_f.barh(y_pos_f, values_f, color=primary_color, align="center")
    ax_f.set_yticks(y_pos_f)
    ax_f.set_yticklabels(labels_f, fontsize=9)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Relative $R^2$ Degradation when Removed ($\Delta R^2$)", fontsize=10)
    ax_f.set_title("(F) Factor Importance Contribution Ranking", fontsize=12, fontweight="bold", pad=10)
    for i, v in enumerate(values_f):
        ax_f.text(v + 0.002, i, f"+{v:.4f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI publication ablation figure to {out_file}")
