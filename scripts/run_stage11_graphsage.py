"""Standalone runner script for Stage 11 — GraphSAGE Architecture Bottleneck Experiment.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graph.sage_trainer import evaluate_graphsage_loso


def append_to_ledger(ledger_path: Path, record: dict) -> None:
    file_exists = ledger_path.exists()
    fieldnames = [
        "Timestamp",
        "Git_Commit",
        "Model",
        "Feature_Set",
        "Seed",
        "Hyperparameters",
        "CV_Strategy",
        "MAE",
        "RMSE",
        "R2",
        "Pearson_r",
        "Pearson_p",
        "Spearman_r",
        "Spearman_p",
        "Median_AE",
        "Explained_Variance",
        "Runtime_sec",
        "Output_Directory",
    ]

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphSAGE Architecture Bottleneck Experiment.")
    parser.add_argument("--config", default="configs/stage11_gcn_regression.yaml")
    parser.add_argument("--device", default="cpu", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage11_graphsage")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    graph_dir = REPOSITORY_ROOT / cfg.get("graph_dataset_dir", "outputs/graph_dataset/wpli_alpha_top20")
    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "outputs/benchmark/stage11")
    ledger_path = REPOSITORY_ROOT / cfg.get("ledger_path", "reports/benchmark_ledger.csv")

    logger.info("==================================================")
    logger.info("Stage 11: PyTorch Geometric GraphSAGE Regressor Benchmark")
    logger.info("Dataset: %s", graph_dir)
    logger.info("==================================================")

    start_time = time.perf_counter()
    res = evaluate_graphsage_loso(graph_dir, cfg, out_dir, device_str=args.device)
    elapsed = time.perf_counter() - start_time

    m = res["metrics"]
    record = {
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Git_Commit": "0cf6132",
        "Model": "graphsage",
        "Feature_Set": "wpli_alpha_top20",
        "Seed": cfg.get("random_seed", 42),
        "Hyperparameters": f"hidden={cfg['model'].get('hidden_channels', 48)},layers={cfg['model'].get('num_layers', 3)},lr={cfg['training']['lr']}",
        "CV_Strategy": "LOSO_109_folds",
        "MAE": round(m["mae"], 6),
        "RMSE": round(m["rmse"], 6),
        "R2": round(m["r2"], 6),
        "Pearson_r": round(m["pearson_r"], 6),
        "Pearson_p": f"{m['pearson_p']:.4e}",
        "Spearman_r": round(m["spearman_r"], 6),
        "Spearman_p": f"{m['spearman_p']:.4e}",
        "Median_AE": round(m["median_ae"], 6),
        "Explained_Variance": round(m["explained_variance"], 6),
        "Runtime_sec": round(elapsed, 2),
        "Output_Directory": str(out_dir),
    }

    append_to_ledger(ledger_path, record)
    logger.info("GraphSAGE Benchmark complete in %.2fs. Ledger updated at %s", elapsed, ledger_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
