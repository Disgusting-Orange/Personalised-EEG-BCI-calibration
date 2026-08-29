#!/usr/bin/env python
"""Master CLI script for Stage 16: Publication-Quality Scientific Ablation Suite.

Executes all 7 scientific ablation experiments using the frozen primary GCN regressor,
generates machine-readable CSV & JSON reports, LaTeX tables, and 300 DPI multi-panel figures.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.graph.ablation_runner import run_all_ablations
from src.graph.ablation_viz import (
    generate_ablation_latex_table,
    generate_publication_ablation_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.run_stage16_ablations")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 16: Publication-Quality Scientific Ablation Suite")
    parser.add_argument("--config", type=str, default="configs/stage16_ablation_studies.yaml", help="Path to Stage 16 YAML config")
    parser.add_argument("--device", type=str, default="cpu", help="Device string ('cpu' or 'cuda')")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = args.device

    out_dir = Path(cfg.get("output_directory", "outputs/ablation_studies"))
    reports_dir = Path(cfg.get("reports_directory", "reports/ablation_studies"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Stage 16: Publication-Quality Scientific Ablation Suite")
    logger.info("==================================================")

    # 1. Run all 7 scientific ablation experiments
    ablation_out = run_all_ablations(cfg)
    df_results = ablation_out["df"]

    # 2. Generate LaTeX Publication Table
    latex_path = reports_dir / "ablation_table.tex"
    generate_ablation_latex_table(df_results, latex_path)

    # 3. Generate 300 DPI Publication Multi-Panel Figure
    fig_path = reports_dir / "ablation_multipanel_figure.png"
    dpi = int(cfg.get("plot_settings", {}).get("dpi", 300))
    generate_publication_ablation_figure(df_results, fig_path, dpi=dpi)

    logger.info("==================================================")
    logger.info("Stage 16 Scientific Ablation Suite Complete.")
    logger.info(f"LaTeX Table: {latex_path}")
    logger.info(f"300 DPI Figure: {fig_path}")
    logger.info(f"CSV Report: {out_dir / 'ablation_results.csv'}")
    logger.info("==================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
