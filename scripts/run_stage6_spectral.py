"""Standalone execution script for Stage 6 — Resting-State Spectral Feature Extraction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from preprocessing.loader import enumerate_subjects
from resting_state.spectral import run_stage6_subject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 Resting-State Spectral Feature Extraction.")
    parser.add_argument("--subjects", nargs="+", help="Explicit subject IDs to run.")
    parser.add_argument("--config", help="Path to Stage 6 configuration file.", default="configs/stage6_spectral_features.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage6")

    config_path = REPOSITORY_ROOT / args.config
    dataset_root = REPOSITORY_ROOT / "data" / "raw" / "eegmmidb"
    discovered = enumerate_subjects(dataset_root)

    subjects = discovered
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]

    logger.info("Running Stage 6 spectral feature extraction for %d subjects", len(subjects))
    passed = 0
    failed = 0

    for idx, subject_id in enumerate(subjects, start=1):
        try:
            result = run_stage6_subject(subject_id, config_path)
            passed += 1
            print(f"[{idx:03d}/{len(subjects)}] {subject_id} PASS (Alpha Blocking Ratio={result.get('alpha_blocking_ratio_occipital', 'N/A')})")
        except Exception as exc:
            failed += 1
            logger.error("Subject %s FAILED Stage 6: %s", subject_id, exc)

    logger.info("Stage 6 complete: %d/%d passed (%d failed)", passed, len(subjects), failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
