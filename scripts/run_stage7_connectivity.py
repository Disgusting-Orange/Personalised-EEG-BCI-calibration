"""Standalone execution script for Stage 7 — Functional Connectivity Framework.

Supports parallel subject processing, checkpoint resume, and comprehensive reporting.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from preprocessing.loader import enumerate_subjects
from resting_state.connectivity import run_stage7_subject, load_config


def _process_subject_wrapper(args_tuple: tuple[str, str, bool, bool]) -> tuple[str, bool, float, str]:
    subject_id, config_path, overwrite, save_heatmaps = args_tuple
    start = time.perf_counter()
    report_file = REPOSITORY_ROOT / "outputs" / "connectivity" / f"stage7_{subject_id.lower()}" / f"{subject_id}_connectivity_report.json"

    if report_file.exists() and not overwrite:
        return subject_id, True, 0.0, "SKIPPED (Resume)"

    try:
        res = run_stage7_subject(subject_id, config_path, save_heatmaps=save_heatmaps)
        elapsed = time.perf_counter() - start
        return subject_id, True, elapsed, "PASS"
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return subject_id, False, elapsed, f"FAIL ({exc})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7 Functional Connectivity Framework.")
    parser.add_argument("--subjects", nargs="+", help="Explicit subject IDs to run.")
    parser.add_argument("--config", help="Path to Stage 7 configuration file.", default="configs/stage7_functional_connectivity.yaml")
    parser.add_argument("--n-jobs", type=int, help="Number of parallel processes.", default=4)
    parser.add_argument("--resume", action="store_true", help="Skip subjects with existing completion reports.")
    parser.add_argument("--save-heatmaps", action="store_true", help="Generate 300 DPI PNG heatmaps for each connectivity matrix.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage7")

    config_path = REPOSITORY_ROOT / args.config
    cfg = load_config(config_path)
    n_jobs = args.n_jobs or int(cfg.get("execution", {}).get("n_jobs", 4))
    save_heatmaps = args.save_heatmaps or bool(cfg.get("execution", {}).get("save_heatmaps", False))

    dataset_root = REPOSITORY_ROOT / "data" / "raw" / "eegmmidb"
    discovered = enumerate_subjects(dataset_root)

    subjects = discovered
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]

    logger.info("Running Stage 7 functional connectivity for %d subjects using n_jobs=%d (resume=%s, save_heatmaps=%s)", len(subjects), n_jobs, args.resume, save_heatmaps)
    
    passed = 0
    failed = 0
    total_start = time.perf_counter()

    tasks = [(subj, str(config_path), not args.resume, save_heatmaps) for subj in subjects]

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sub = {executor.submit(_process_subject_wrapper, t): t[0] for t in tasks}
            for idx, future in enumerate(as_completed(future_to_sub), start=1):
                sub_id, ok, elapsed, msg = future.result()
                if ok:
                    passed += 1
                else:
                    failed += 1
                print(f"[{idx:03d}/{len(subjects)}] {sub_id} {msg} ({elapsed:.2f}s)")
    else:
        for idx, t in enumerate(tasks, start=1):
            sub_id, ok, elapsed, msg = _process_subject_wrapper(t)
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"[{idx:03d}/{len(subjects)}] {sub_id} {msg} ({elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - total_start
    logger.info("Stage 7 complete: %d/%d passed (%d failed) in %.2fs", passed, len(subjects), failed, total_elapsed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
