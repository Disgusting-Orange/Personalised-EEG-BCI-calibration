"""Standalone execution script for Stage 10 — Graph Dataset Construction (PyTorch Geometric).

Builds PyG Data objects for specified subjects, validates graph integrity across 3 phases,
and collates PyTorch Geometric InMemoryDataset structures.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from src.graph.dataset import EEGGraphDataset
from src.graph.validator import validate_graph_dataset_directory
from preprocessing.loader import enumerate_subjects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 10 Graph Dataset Construction (PyTorch Geometric).")
    parser.add_argument("--subjects", nargs="+", help="Explicit subject IDs to construct.")
    parser.add_argument("--config", help="Path to Stage 10 configuration file.", default="configs/stage10_graph_dataset.yaml")
    parser.add_argument("--n-jobs", type=int, help="Parallel processing workers.", default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from existing graph files.")
    parser.add_argument("--phase", choices=["1", "2", "3"], help="Explicit validation phase (1: S001-S003, 2: S001-S010, 3: Full cohort).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage10")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    n_jobs = args.n_jobs or int(cfg.get("n_jobs", 8))
    conn_metric = cfg.get("connectivity_metric", "wpli")
    freq_band = cfg.get("frequency_band", "alpha")
    density = float(cfg.get("sparsification_density", 0.15))
    density_str = f"top{int(density*100)}" if density < 1.0 else "fully_connected"

    feat_type = cfg.get("node_feature_type", "concatenated")
    feat_str = f"_{feat_type}" if feat_type != "concatenated" else ""
    dataset_name = f"{conn_metric}_{freq_band}{feat_str}_{density_str}"
    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "outputs/graph_dataset") / dataset_name

    dataset_root = REPOSITORY_ROOT / "data" / "raw" / "eegmmidb"
    discovered = enumerate_subjects(dataset_root)

    subjects = discovered
    if args.phase == "1":
        subjects = ["S001", "S002", "S003"]
    elif args.phase == "2":
        subjects = [f"S{i:03d}" for i in range(1, 11)]
    elif args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]

    logger.info("==================================================")
    logger.info("Stage 10: Graph Dataset Construction (%s)", dataset_name)
    logger.info("Subjects: %d | n_jobs: %d | Density: %.2f", len(subjects), n_jobs, density)
    logger.info("==================================================")

    start_time = time.perf_counter()

    ds = EEGGraphDataset(
        root=out_dir,
        subjects=subjects,
        config=cfg,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Graph dataset construction complete for %d subjects in %.2fs", len(ds), elapsed)

    # Validate Graph Dataset
    v_res = validate_graph_dataset_directory(out_dir, expected_subjects=len(subjects))
    logger.info("Validation Status: %s (%d/%d graphs passed)", v_res["overall_status"], v_res["passed_graphs"], v_res["total_graphs_evaluated"])

    return 0 if v_res["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
