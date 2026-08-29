"""Standalone runner script for Stage 15 — Master Publication Suite & LaTeX Table Generator.

Generates 4-panel 300 DPI master publication figures, LaTeX comparison tables,
and comprehensive publication audit reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graph.publication_suite import generate_latex_table, generate_master_figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 15 Master Publication Suite.")
    parser.add_argument("--config", default="configs/stage15_publication_suite.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage15")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ledger_path = REPOSITORY_ROOT / cfg.get("ledger_path", "reports/benchmark_ledger.csv")
    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "reports/publication_suite")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Stage 15: Master Publication Suite Generator")
    logger.info("Ledger: %s", ledger_path)
    logger.info("==================================================")

    start_time = time.perf_counter()

    df_ledger = pd.read_csv(ledger_path)
    ablation_path = REPOSITORY_ROOT / "outputs/statistical_analysis/topology_ablation.csv"
    df_ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()

    # 1. Generate LaTeX Table Code
    latex_code = generate_latex_table(df_ledger)
    (out_dir / "master_comparison_table.tex").write_text(latex_code, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", out_dir / "master_comparison_table.tex")

    # 2. Generate 4-Panel Master Publication Figure
    generate_master_figure(df_ledger, df_ablation, out_dir, dpi=int(cfg.get("dpi", 300)))

    elapsed = time.perf_counter() - start_time

    val_report = {
        "status": "PASS",
        "total_models_summarized": len(df_ledger),
        "latex_table_generated": True,
        "master_figure_generated": True,
        "elapsed_seconds": round(elapsed, 2),
    }

    (out_dir / "validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
    logger.info("Stage 15 Master Publication Suite complete in %.2fs. Report saved at %s", elapsed, out_dir / "validation_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
