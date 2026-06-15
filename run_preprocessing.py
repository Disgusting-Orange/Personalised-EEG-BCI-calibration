"""
run_preprocessing.py
--------------------
Entry-point script for the EEG preprocessing pipeline.

Usage examples
--------------
# Full dataset
python run_preprocessing.py

# Single subject
python run_preprocessing.py --subjects S001

# Multiple subjects, specific runs (motor imagery only)
python run_preprocessing.py --subjects S001 S002 S003 --runs R03 R04 R07 R08

# All subjects, save as numpy instead of FIF
python run_preprocessing.py --format numpy

# Force reprocess (overwrite existing outputs)
python run_preprocessing.py --overwrite

# Dry-run: just print what would be processed
python run_preprocessing.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# ── Make the preprocess package importable when running from this directory ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import config as cfg
from preprocess.pipeline import preprocess_all
from preprocess.utils import ensure_dirs, setup_logging


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet EEG Motor Imagery — Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--subjects", nargs="*", default=None,
        metavar="SXXX",
        help="Subject IDs to process (e.g. S001 S002). Default: all subjects.",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        metavar="RXX",
        help="Run IDs to process (e.g. R03 R04). Default: all runs.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=cfg.DATA_DIR,
        help=f"Root data directory. Default: {cfg.DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=cfg.OUTPUT_DIR,
        help=f"Preprocessed output directory. Default: {cfg.OUTPUT_DIR}",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=cfg.FIGURE_DIR,
        help=f"Figures output directory. Default: {cfg.FIGURE_DIR}",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=cfg.LOG_DIR,
        help=f"Log directory. Default: {cfg.LOG_DIR}",
    )
    parser.add_argument(
        "--format", choices=["fif", "numpy"], default=cfg.SAVE_FORMAT,
        dest="save_format",
        help="Output file format (fif = MNE Epochs, numpy = .npz). Default: fif.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=cfg.OVERWRITE,
        help="Overwrite existing preprocessed files.",
    )
    parser.add_argument(
        "--no-figures", action="store_true", default=False,
        help="Skip saving visualisation figures (faster).",
    )
    parser.add_argument(
        "--show-figures", action="store_true", default=False,
        help="Display figures interactively (for Jupyter / local use).",
    )
    parser.add_argument(
        "--notch-freq", type=float, default=cfg.NOTCH_FREQ,
        help=f"Powerline notch frequency in Hz. Default: {cfg.NOTCH_FREQ}.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=cfg.N_JOBS,
        help=f"Parallel jobs for MNE (-1 = all cores). Default: {cfg.N_JOBS}.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print the list of files that would be processed and exit.",
    )
    parser.add_argument(
        "--summary-json", type=Path, default=None,
        metavar="PATH",
        help="If given, write the pipeline summary dict to this JSON file.",
    )

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_cli_overrides(args: argparse.Namespace) -> None:
    """Push CLI arguments back into the config module at runtime."""
    cfg.SAVE_FORMAT   = args.save_format
    cfg.OVERWRITE     = args.overwrite
    cfg.NOTCH_FREQ    = args.notch_freq
    cfg.N_JOBS        = args.n_jobs
    cfg.SAVE_FIGURES  = not args.no_figures
    cfg.SHOW_FIGURES  = args.show_figures


def dry_run(args: argparse.Namespace) -> None:
    """Print EDF files that would be processed."""
    from preprocess.loader import discover_subject_runs
    files = discover_subject_runs(
        args.data_dir,
        subjects=args.subjects,
        runs=args.runs,
    )
    print(f"\nDry run — {len(files)} file(s) would be processed:\n")
    for f in files:
        print(f"  {f.relative_to(args.data_dir)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    apply_cli_overrides(args)

    # Create output directories
    ensure_dirs(args.output_dir, args.figure_dir, args.log_dir)

    logger = setup_logging(args.log_dir)
    logger.info("=" * 60)
    logger.info("PhysioNet EEG Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data dir   : {args.data_dir}")
    logger.info(f"Output dir : {args.output_dir}")
    logger.info(f"Format     : {args.save_format}")
    logger.info(f"Overwrite  : {args.overwrite}")
    logger.info(f"Subjects   : {args.subjects or 'ALL'}")
    logger.info(f"Runs       : {args.runs or 'ALL'}")

    if args.dry_run:
        dry_run(args)
        return

    # Run pipeline
    summary = preprocess_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        log_dir=args.log_dir,
        subjects=args.subjects,
        runs=args.runs,
    )

    # Persist summary
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as fh:
            json.dump(summary["summary"], fh, indent=2)
        logger.info(f"Summary written to {args.summary_json}")

    # Print final stats
    s = summary["summary"]
    print(
        f"\n{'='*50}\n"
        f"Pipeline finished\n"
        f"  Success      : {s.get('success', 0)}\n"
        f"  Skipped      : {s.get('skipped', 0)}\n"
        f"  Failed       : {s.get('failed', 0)}\n"
        f"  Total epochs : {s.get('total_epochs', 0)}\n"
        f"{'='*50}\n"
    )


if __name__ == "__main__":
    main()
