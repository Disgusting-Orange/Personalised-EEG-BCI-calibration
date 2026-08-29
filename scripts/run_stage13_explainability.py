"""Standalone execution script for Stage 13 — Publication-Quality Graph Explainability.

Extracts cohort-wide electrode node importances, functional connectivity edge masks,
spectral band contributions, scalp topomap figures, and GCN vs GAT alignment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graph.explainability import explain_cohort
from graph.explainability_viz import plot_edge_matrix, plot_gcn_vs_gat_correlation, plot_scalp_topomap
from preprocessing.loader import enumerate_subjects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 13 Graph Explainability (GNNExplainer).")
    parser.add_argument("--config", default="configs/stage13_explainability.yaml")
    parser.add_argument("--epochs", type=int, default=150, help="GNNExplainer optimization epochs.")
    parser.add_argument("--phase", choices=["1", "2", "3"], help="Validation phase (1: S001-S003, 2: S001-S010, 3: Full cohort).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage13")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    graph_dir = REPOSITORY_ROOT / cfg.get("graph_dataset_dir", "outputs/graph_dataset/wpli_alpha_top15")
    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "outputs/explainability")

    dataset_root = REPOSITORY_ROOT / "data" / "raw" / "eegmmidb"
    discovered = enumerate_subjects(dataset_root)

    subjects = discovered
    if args.phase == "1":
        subjects = ["S001", "S002", "S003"]
    elif args.phase == "2":
        subjects = [f"S{i:03d}" for i in range(1, 11)]

    logger.info("==================================================")
    logger.info("Stage 13: Publication-Quality Graph Explainability")
    logger.info("Subjects: %d | Explainer Epochs: %d", len(subjects), args.epochs)
    logger.info("==================================================")

    start_time = time.perf_counter()

    # 1. Explain GCN
    res_gcn = explain_cohort("gcn", graph_dir, out_dir, subjects=subjects, epochs=args.epochs)
    plot_scalp_topomap(res_gcn["df_nodes"], "gcn", out_dir / "gcn" / "gcn_scalp_topomap.png")
    plot_edge_matrix(res_gcn["mean_edge_matrix"], "gcn", out_dir / "gcn" / "gcn_edge_matrix.png")

    # 2. Explain GAT
    res_gat = explain_cohort("gat", graph_dir, out_dir, subjects=subjects, epochs=args.epochs)
    plot_scalp_topomap(res_gat["df_nodes"], "gat", out_dir / "gat" / "gat_scalp_topomap.png")
    plot_edge_matrix(res_gat["mean_edge_matrix"], "gat", out_dir / "gat" / "gat_edge_matrix.png")

    # 3. GCN vs GAT Correlation Alignment
    corr = plot_gcn_vs_gat_correlation(res_gcn["df_nodes"], res_gat["df_nodes"], out_dir / "gcn_vs_gat_explainability_correlation.png")
    logger.info("GCN vs GAT Node Importance Alignment: Pearson r=%.4f (p=%.4e), Spearman ρ=%.4f (p=%.4e)", corr["pearson_r"], corr["pearson_p"], corr["spearman_r"], corr["spearman_p"])

    elapsed = time.perf_counter() - start_time

    val_report = {
        "status": "PASS",
        "phase": args.phase or "3",
        "total_subjects": len(subjects),
        "gcn_top_nodes": res_gcn["df_nodes"].head(5)["channel"].tolist(),
        "gat_top_nodes": res_gat["df_nodes"].head(5)["channel"].tolist(),
        "gcn_vs_gat_correlation": corr,
        "elapsed_seconds": round(elapsed, 2),
    }

    (out_dir / "validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
    logger.info("Stage 13 Explainability complete in %.2fs. Report saved at %s", elapsed, out_dir / "validation_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
