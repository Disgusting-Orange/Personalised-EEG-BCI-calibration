"""Publication-quality graphics renderer for Stage 13 GNN Explainability.

Renders 300 DPI scalp electrode topomaps, functional connectivity edge heatmaps,
spectral band contribution bar plots, and GCN vs GAT correlation plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

logger = logging.getLogger("graph.explainability_viz")

# 10-05 Scalp Coordinates (Approximate 2D projections)
SCALP_COORDS_2D = {
    "Fp1": (-0.3, 0.9), "Fp2": (0.3, 0.9), "F7": (-0.7, 0.6), "F3": (-0.4, 0.6),
    "Fz": (0.0, 0.6), "F4": (0.4, 0.6), "F8": (0.7, 0.6), "AF3": (-0.2, 0.75),
    "AF4": (0.2, 0.75), "AF7": (-0.5, 0.75), "AF8": (0.5, 0.75), "F1": (-0.2, 0.6),
    "F2": (0.2, 0.6), "F5": (-0.55, 0.6), "F6": (0.55, 0.6), "FC5": (-0.6, 0.35),
    "FC3": (-0.4, 0.35), "FC1": (-0.2, 0.35), "FCz": (0.0, 0.35), "FC2": (0.2, 0.35),
    "FC4": (0.4, 0.35), "FC6": (0.6, 0.35), "C5": (-0.7, 0.0), "C3": (-0.4, 0.0),
    "C1": (-0.2, 0.0), "Cz": (0.0, 0.0), "C2": (0.2, 0.0), "C4": (0.4, 0.0),
    "C6": (0.7, 0.0), "CP5": (-0.6, -0.35), "CP3": (-0.4, -0.35), "CP1": (-0.2, -0.35),
    "CPz": (0.0, -0.35), "CP2": (0.2, -0.35), "CP4": (0.4, -0.35), "CP6": (0.6, -0.35),
    "P5": (-0.55, -0.6), "P3": (-0.4, -0.6), "P1": (-0.2, -0.6), "Pz": (0.0, -0.6),
    "P2": (0.2, -0.6), "P4": (0.4, -0.6), "P6": (0.55, -0.6), "PO7": (-0.5, -0.75),
    "PO3": (-0.2, -0.75), "POz": (0.0, -0.75), "PO4": (0.2, -0.75), "PO8": (0.5, -0.75),
    "O1": (-0.3, -0.9), "Oz": (0.0, -0.9), "O2": (0.3, -0.9), "CB1": (-0.4, -1.0),
    "CB2": (0.4, -1.0), "FT7": (-0.8, 0.35), "FT8": (0.8, 0.35), "T7": (-0.9, 0.0),
    "T8": (0.9, 0.0), "TP7": (-0.8, -0.35), "TP8": (0.8, -0.35), "T9": (-1.0, 0.0),
    "T10": (1.0, 0.0), "Iz": (0.0, -1.0)
}


def plot_scalp_topomap(
    df_nodes: pd.DataFrame,
    model_id: str,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Render 300 DPI publication scalp electrode importance scatter topomap."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)

    # Draw head outline
    head_circle = plt.Circle((0, 0), 1.0, color="k", fill=False, linewidth=2.0)
    nose = plt.Polygon([[-0.1, 1.0], [0.0, 1.15], [0.1, 1.0]], color="k", fill=False, linewidth=2.0)
    ax.add_patch(head_circle)
    ax.add_patch(nose)

    # Extract coordinates & importances
    xs, ys, s_sizes, colors = [], [], [], []

    coords_map = {k.lower(): v for k, v in SCALP_COORDS_2D.items()}
    max_imp = df_nodes["importance"].max() if not df_nodes.empty else 1.0
    for _, row in df_nodes.iterrows():
        ch = str(row["channel"])
        imp = row["importance"]
        ch_lower = ch.lower()
        if ch_lower in coords_map:
            x, y = coords_map[ch_lower]
            xs.append(x)
            ys.append(y)
            s_sizes.append(100 + 400 * (imp / max_imp))
            colors.append(imp)
            ax.annotate(ch, (x, y), fontsize=8, fontweight="bold", ha="center", va="center", color="white")

    sc = ax.scatter(xs, ys, c=colors, s=s_sizes, cmap="plasma", edgecolors="k", linewidth=1.0, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Node Importance Weight", fontsize=10, fontweight="bold")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Stage 13: {model_id.upper()} Electrode Importance Topography", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)


def plot_edge_matrix(
    edge_matrix: np.ndarray,
    model_id: str,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> None:
    """Render 64x64 functional connectivity edge importance heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    sns.heatmap(edge_matrix, cmap="magma", ax=ax, cbar_kws={"label": "Edge Importance Weight"})
    ax.set_title(f"Stage 13: {model_id.upper()} Edge Importance Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Electrode Index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Electrode Index", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)


def plot_gcn_vs_gat_correlation(
    df_gcn_nodes: pd.DataFrame,
    df_gat_nodes: pd.DataFrame,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> dict[str, float]:
    """Scatter plot and correlation evaluation comparing GCN vs GAT node importances."""
    merged = pd.merge(df_gcn_nodes, df_gat_nodes, on="channel", suffixes=("_gcn", "_gat"))

    x = merged["importance_gcn"].values
    y = merged["importance_gat"].values

    p_res = stats.pearsonr(x, y)
    s_res = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    ax.scatter(x, y, color="#9467bd", alpha=0.8, edgecolors="k", linewidth=0.5, s=60)

    for _, row in merged.iterrows():
        ax.annotate(row["channel"], (row["importance_gcn"], row["importance_gat"]), fontsize=7, alpha=0.7)

    # Linear fit
    slope, intercept, _, _, _ = stats.linregress(x, y)
    min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], "r--", linewidth=1.5, label="Linear Fit")

    ax.set_xlabel("GCN Node Importance Weight", fontsize=11, fontweight="bold")
    ax.set_ylabel("GAT Node Importance Weight", fontsize=11, fontweight="bold")
    ax.set_title("Stage 13: GCN vs. GAT Node Importance Alignment", fontsize=12, fontweight="bold", pad=12)

    textstr = f"Pearson r: {p_res.statistic:.4f} (p={p_res.pvalue:.4e})\nSpearman ρ: {s_res.statistic:.4f} (p={s_res.pvalue:.4e})"
    props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9, edgecolor="grey")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi)
    plt.savefig(out_p.with_suffix(".svg"))
    plt.close(fig)

    return {
        "pearson_r": float(p_res.statistic),
        "pearson_p": float(p_res.pvalue),
        "spearman_r": float(s_res.statistic),
        "spearman_p": float(s_res.pvalue),
    }
