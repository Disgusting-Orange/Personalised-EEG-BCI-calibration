"""Batch checkpoint execution framework for Stages 2-5.

Orchestration ONLY. This script does not implement any scientific logic.
It invokes existing frozen pipeline functions and stage entry points to run
the per-subject pipeline:

    Stage 2 (preprocessing, all baseline runs)
      -> Stage 3 (MI decoding target generation)
      -> Stage 4 (target quality validation)
      -> Stage 5 (resting-state validation)

Design rules:
  * No new third-party dependencies (Python standard library only, plus
    PyYAML which is already a project dependency).
  * Frozen configs under configs/ are NEVER modified. Per-subject runtime
    configs are generated in the checkpoint run directory.
  * Frozen pipeline functions are invoked unchanged.
  * Subject discovery uses the frozen enumerate_subjects() helper.
  * A failed stage marks the subject FAILED, captures the full traceback,
    skips the remaining stages for that subject, and continues with the next.
  * Every execution writes a new timestamped directory under
    outputs/checkpoint_runs/ — previous runs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from preprocessing.artifacts import (  # noqa: E402
    detect_artifact_components,
    detect_bad_channels,
    run_bad_channel_pipeline,
    run_ica_pipeline,
)
from preprocessing.epochs import run_epoch_pipeline  # noqa: E402
from preprocessing.filters import run_filter_pipeline  # noqa: E402
from preprocessing.loader import enumerate_subjects  # noqa: E402
from preprocessing.qc import run_qc_pipeline  # noqa: E402

SUCCESS_LEVEL = 25  # between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = _success  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Stage runners
#
# Each function executes ONE stage for ONE subject and returns a dict with
# at least {"ok": bool, "runtime_seconds": float}. On failure it returns
# {"ok": False, "error": ..., "traceback": ...} without raising.
# ---------------------------------------------------------------------------


def _run_stage2(subject_id: str, logger: logging.Logger) -> dict:
    """Stage 2: preprocess all baseline runs (R01 eyes-open, R02 eyes-closed).

    Invokes the frozen run_*_pipeline functions directly, mirroring the
    established pattern in scripts/run_stage2_s001_r02.py.
    """
    config = _load_yaml(REPOSITORY_ROOT / "configs" / "preprocessing.yaml")
    run_mapping = _load_yaml(REPOSITORY_ROOT / "configs" / "eegmmidb_run_mapping.yaml")
    baseline_runs = [
        run_id
        for run_id, info in run_mapping.get("run_mapping", {}).items()
        if info.get("role") == "baseline"
    ]
    if not baseline_runs:
        return {"ok": False, "error": "No baseline runs found in run-mapping config"}

    preprocessed_root = REPOSITORY_ROOT / "outputs" / "preprocessed"
    dataset_root = REPOSITORY_ROOT / config["dataset_root"]

    for run_id in baseline_runs:
        edf_path = dataset_root / subject_id / f"{subject_id}{run_id}.edf"
        if not edf_path.exists():
            return {"ok": False, "error": f"Missing EDF for {subject_id}/{run_id}: {edf_path}"}

        logger.info("Stage 2: %s/%s filtering", subject_id, run_id)
        filtered_raw = run_filter_pipeline(
            edf_path,
            subject=subject_id,
            run=run_id,
            config=config,
            save_filtered=True,
            output_dir=preprocessed_root / "filtered",
            overwrite=True,
        )

        bad_channels = detect_bad_channels(filtered_raw, config=config)
        bad_channel_raw = run_bad_channel_pipeline(
            filtered_raw,
            subject=subject_id,
            run=run_id,
            config=config,
            save_interpolated=False,
            output_dir=preprocessed_root,
            overwrite=True,
        )

        logger.info("Stage 2: %s/%s ICA", subject_id, run_id)
        ica_raw, ica_object = run_ica_pipeline(
            bad_channel_raw,
            subject=subject_id,
            run=run_id,
            config=config,
            save_ica_model=True,
            output_dir=preprocessed_root / "ica",
            overwrite=True,
        )
        ica_components_removed: list[int] = []
        if ica_object is not None:
            ica_components_removed = detect_artifact_components(
                ica_object, bad_channel_raw, config=config
            )

        logger.info("Stage 2: %s/%s epoching", subject_id, run_id)
        epochs, _ = run_epoch_pipeline(
            ica_raw,
            subject=subject_id,
            run=run_id,
            config=config,
            save_epochs_output=True,
            output_dir=preprocessed_root / "epochs",
            overwrite=True,
        )

        logger.info("Stage 2: %s/%s QC", subject_id, run_id)
        run_qc_pipeline(
            raw=ica_raw,
            epochs=epochs,
            subject=subject_id,
            run=run_id,
            bad_channels=bad_channels,
            ica_components_removed=ica_components_removed,
            config=config,
            output_dir=preprocessed_root / "qc",
        )

    return {"ok": True}


def _run_stage3(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 3: generate the MI decoding target for one subject.

    Builds a per-subject runtime config (copy of the frozen Stage 3 config
    with subject_id and output_directory parameterized) and invokes the
    frozen run_stage3_target_generation entry point. The frozen config file
    under configs/ is never modified.
    """
    from mi_decoding import run_stage3_target_generation

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage3_mi_decoder.yaml")
    runtime_config = deepcopy(frozen_config)
    subject_output_dir = REPOSITORY_ROOT / "outputs" / "targets" / f"stage3_{subject_id.lower()}"
    runtime_config["subject_id"] = subject_id
    runtime_config["output_directory"] = str(subject_output_dir)

    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage3.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 3: %s target generation (config=%s)", subject_id, runtime_config_path)
    run_stage3_target_generation(runtime_config_path)
    return {"ok": True}


def _run_stage4(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 4: validate the subject's target quality.

    Builds a per-subject runtime config pointing at the subject's Stage 3
    output directory. The frozen config under configs/ is never modified.
    """
    from target_quality import run_stage4

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage4_target_quality.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config["target_directory"] = str(
        REPOSITORY_ROOT / "outputs" / "targets" / f"stage3_{subject_id.lower()}"
    )
    runtime_config["output_directory"] = str(
        REPOSITORY_ROOT / "reports" / f"stage4_target_quality_{subject_id.lower()}"
    )

    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage4.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 4: %s target validation (config=%s)", subject_id, runtime_config_path)
    run_stage4(runtime_config_path)
    return {"ok": True}


def _run_stage5(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 5: validate the subject's resting-state preprocessing.

    Builds a per-subject runtime config scoped to the single subject. The
    frozen config under configs/ is never modified.
    """
    from resting_state import run_stage5

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage5_resting_state_validation.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config["subjects"] = [subject_id]
    runtime_config["output_directory"] = str(
        REPOSITORY_ROOT / "reports" / f"stage5_resting_state_validation_{subject_id.lower()}"
    )

    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage5.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 5: %s resting-state validation (config=%s)", subject_id, runtime_config_path)
    run_stage5(runtime_config_path)
    return {"ok": True}


def _run_stage6(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 6: extract resting-state spectral features for the subject."""
    from resting_state.spectral import run_stage6_subject

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage6_spectral_features.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage6.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 6: %s spectral feature extraction (config=%s)", subject_id, runtime_config_path)
    res = run_stage6_subject(subject_id, runtime_config_path)
    return {"ok": res.get("status") == "PASS"}


def _run_stage7(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 7: compute resting-state functional connectivity for the subject."""
    from resting_state.connectivity import run_stage7_subject

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage7_functional_connectivity.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage7.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 7: %s functional connectivity (config=%s)", subject_id, runtime_config_path)
    res = run_stage7_subject(subject_id, runtime_config_path)
    return {"ok": res.get("status") == "PASS"}


def _run_stage8(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 8: classical baseline regressor evaluation."""
    from baselines.evaluator import evaluate_model_loso
    from baselines.feature_loader import load_dataset

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage8_classical_baselines.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage8.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 8: %s classical baseline regressor evaluation (config=%s)", subject_id, runtime_config_path)
    X, y, subject_ids, feature_names = load_dataset(feature_set="spectral_concatenated")
    res = evaluate_model_loso("ridge", X, y, feature_names, subject_ids, runtime_config)
    return {"ok": res.get("metrics") is not None}


def _run_stage10(subject_id: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Stage 10: PyTorch Geometric graph dataset construction."""
    from graph.builder import build_subject_graph
    from graph.validator import validate_graph

    frozen_config = _load_yaml(REPOSITORY_ROOT / "configs" / "stage10_graph_dataset.yaml")
    runtime_config = deepcopy(frozen_config)
    runtime_config_path = run_dir / "runtime_configs" / f"{subject_id}_stage10.yaml"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_config, fh, sort_keys=False)

    logger.info("Stage 10: %s graph dataset construction (config=%s)", subject_id, runtime_config_path)
    data = build_subject_graph(subject_id, runtime_config)
    res = validate_graph(data)
    return {"ok": res.get("status") == "PASS"}


STAGE_FUNCS = {
    "Stage2": _run_stage2,
    "Stage3": _run_stage3,
    "Stage4": _run_stage4,
    "Stage5": _run_stage5,
    "Stage6": _run_stage6,
    "Stage7": _run_stage7,
    "Stage8": _run_stage8,
    "Stage10": _run_stage10,
}


# ---------------------------------------------------------------------------
# Per-subject pipeline execution
# ---------------------------------------------------------------------------


def _execute_subject(
    subject_id: str,
    index: int,
    total: int,
    run_dir: Path,
    logger: logging.Logger,
    stop_on_error: bool,
) -> dict:
    """Run Stage 2 -> 3 -> 4 -> 5 for one subject. Never raises."""
    logger.info("=" * 50)
    logger.info("SUBJECT %s", subject_id)
    logger.info("=" * 50)

    subject_start = time.perf_counter()
    stage_runtimes: dict[str, float] = {}
    status = "PASS"
    failed_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    last_success_stage: str | None = None

    for stage_name, stage_func in STAGE_FUNCS.items():
        stage_start = time.perf_counter()
        logger.info("%s started", stage_name)
        try:
            if stage_name in ("Stage3", "Stage4", "Stage5", "Stage6", "Stage7", "Stage8", "Stage10"):
                result = stage_func(subject_id, run_dir, logger)
            else:
                result = stage_func(subject_id, logger)
        except Exception as exc:
            runtime = time.perf_counter() - stage_start
            stage_runtimes[stage_name] = runtime
            tb = traceback.format_exc()
            logger.error("%s FAILED", stage_name)
            logger.error("Exception Type: %s", type(exc).__name__)
            logger.error("Exception Message: %s", exc)
            logger.error("Complete traceback:\n%s", tb)
            logger.error("Last successful stage: %s", last_success_stage or "none")
            status = "FAIL"
            failed_stage = stage_name
            error_type = type(exc).__name__
            error_message = str(exc)
            error_traceback = tb
            break

        runtime = time.perf_counter() - stage_start
        stage_runtimes[stage_name] = runtime
        if not result.get("ok", False):
            logger.error("%s FAILED", stage_name)
            logger.error("Error: %s", result.get("error", "unknown"))
            logger.error("Last successful stage: %s", last_success_stage or "none")
            status = "FAIL"
            failed_stage = stage_name
            error_type = "StageError"
            error_message = str(result.get("error", "unknown"))
            error_traceback = ""
            break

        logger.success("%s complete (%.2fs)", stage_name, runtime)
        last_success_stage = stage_name

    subject_runtime = time.perf_counter() - subject_start

    logger.info("-" * 50)
    if status == "PASS":
        logger.success(
            "SUBJECT %s PASS (%.2f seconds)", subject_id, subject_runtime
        )
    else:
        logger.error(
            "SUBJECT %s FAIL at %s (%.2f seconds)", subject_id, failed_stage, subject_runtime
        )
    logger.info("-" * 50)

    return {
        "subject_id": subject_id,
        "stage_runtimes": stage_runtimes,
        "stage_status": {
            s: ("PASS" if stage_runtimes.get(s, None) is not None and failed_stage != s
                else ("FAIL" if failed_stage == s else "SKIPPED"))
            for s in STAGE_FUNCS
        },
        "subject_runtime": subject_runtime,
        "status": status,
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_summary_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Subject", "Stage2", "Stage3", "Stage4", "Stage5", "Runtime", "Status", "Error"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Subject": r["subject_id"],
                "Stage2": r["stage_status"].get("Stage2", "SKIPPED"),
                "Stage3": r["stage_status"].get("Stage3", "SKIPPED"),
                "Stage4": r["stage_status"].get("Stage4", "SKIPPED"),
                "Stage5": r["stage_status"].get("Stage5", "SKIPPED"),
                "Runtime": f"{r['subject_runtime']:.2f}",
                "Status": r["status"],
                "Error": r.get("error_message") or "",
            })


def _write_failures_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Subject", "Failed Stage", "Exception Type", "Exception Message"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r["status"] == "FAIL":
                writer.writerow({
                    "Subject": r["subject_id"],
                    "Failed Stage": r.get("failed_stage", ""),
                    "Exception Type": r.get("error_type", ""),
                    "Exception Message": r.get("error_message", ""),
                })


def _write_summary_md(
    results: list[dict],
    run_dir: Path,
    total_runtime: float,
    discovered_count: int,
    logger: logging.Logger,
) -> None:
    completed = [r for r in results if r["status"] in ("PASS", "FAIL")]
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]

    runtimes = [r["subject_runtime"] for r in completed]
    avg_runtime = statistics.mean(runtimes) if runtimes else 0.0
    median_runtime = statistics.median(runtimes) if runtimes else 0.0
    fastest = min(completed, key=lambda r: r["subject_runtime"]) if completed else None
    slowest = max(completed, key=lambda r: r["subject_runtime"]) if completed else None

    per_stage_cumulative: dict[str, float] = {}
    for stage in STAGE_FUNCS:
        per_stage_cumulative[stage] = sum(
            r["stage_runtimes"].get(stage, 0.0) for r in completed
        )

    lines: list[str] = []
    lines.append("# Checkpoint Run Summary")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Run timestamp: `{run_dir.name}`")
    lines.append(f"- Total runtime: {total_runtime:.2f} seconds")
    lines.append(f"- Subjects discovered: {discovered_count}")
    lines.append(f"- Subjects completed: {len(completed)}")
    lines.append(f"- Subjects passed: {len(passed)}")
    lines.append(f"- Subjects failed: {len(failed)}")
    lines.append(f"- Average runtime: {avg_runtime:.2f} seconds")
    lines.append(f"- Median runtime: {median_runtime:.2f} seconds")
    if fastest:
        lines.append(f"- Fastest subject: {fastest['subject_id']} ({fastest['subject_runtime']:.2f}s)")
    if slowest:
        lines.append(f"- Slowest subject: {slowest['subject_id']} ({slowest['subject_runtime']:.2f}s)")
    lines.append("")
    lines.append("## Per-stage cumulative runtime")
    lines.append("")
    lines.append("| Stage | Cumulative runtime (s) |")
    lines.append("|---|---|")
    for stage, rt in per_stage_cumulative.items():
        lines.append(f"| {stage} | {rt:.2f} |")
    lines.append("")

    if failed:
        lines.append("## Failure table")
        lines.append("")
        lines.append("| Subject | Failed stage | Exception | Message |")
        lines.append("|---|---|---|---|")
        for r in failed:
            msg = (r.get("error_message") or "").replace("|", "\\|")
            lines.append(
                f"| {r['subject_id']} | {r.get('failed_stage', '')} | "
                f"{r.get('error_type', '')} | {msg} |"
            )
        lines.append("")

    warnings = [r for r in results if r["status"] == "WARNING"]
    if warnings:
        lines.append("## Warning table")
        lines.append("")
        lines.append("| Subject | Note |")
        lines.append("|---|---|")
        for r in warnings:
            lines.append(f"| {r['subject_id']} | {r.get('error_message', '')} |")
        lines.append("")

    summary_path = run_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote summary.md to %s", summary_path)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _find_latest_run_dir(checkpoint_root: Path) -> Path | None:
    if not checkpoint_root.exists():
        return None
    run_dirs = sorted(
        [d for d in checkpoint_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _load_passed_subjects(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    passed: set[str] = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Status") == "PASS":
                passed.add(row["Subject"])
    return passed


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch checkpoint execution for Stages 2-5.")
    parser.add_argument("--resume", action="store_true", help="Skip subjects already marked PASS.")
    parser.add_argument("--subjects", nargs="+", help="Explicit subject IDs to run.")
    parser.add_argument("--start", help="Start from this subject ID (inclusive).")
    parser.add_argument("--stop-on-error", action="store_true", help="Terminate batch on first failure.")
    args = parser.parse_args(argv)

    # Configure logging to console first (file handler added after run_dir
    # is created, so resume can find the latest PRIOR run directory).
    logger = logging.getLogger("checkpoint")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    # Subject discovery via frozen helper (no hardcoding).
    dataset_root = REPOSITORY_ROOT / "data" / "raw" / "eegmmidb"
    discovered = enumerate_subjects(dataset_root)
    logger.info("Discovered %d subjects", len(discovered))

    # Apply filters.
    subjects = discovered
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    if args.start:
        try:
            idx = subjects.index(args.start)
            subjects = subjects[idx:]
        except ValueError:
            logger.error("--start subject %s not found in discovered set", args.start)
            return 2

    # Resume: skip subjects already marked PASS in the latest prior run.
    # This must happen BEFORE creating the new run directory, otherwise the
    # new empty dir is found as "latest" and no subjects are skipped.
    if args.resume:
        latest = _find_latest_run_dir(REPOSITORY_ROOT / "outputs" / "checkpoint_runs")
        if latest is not None:
            already_passed = _load_passed_subjects(latest / "summary.csv")
            if already_passed:
                logger.info("Resume: skipping %d already-PASS subjects from %s", len(already_passed), latest.name)
                subjects = [s for s in subjects if s not in already_passed]

    if not subjects:
        logger.warning("No subjects to run after filtering.")
        return 0

    # Fresh timestamped run directory — created AFTER resume filtering so
    # it does not shadow the latest prior run directory.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = REPOSITORY_ROOT / "outputs" / "checkpoint_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler now that run_dir exists.
    file_handler = logging.FileHandler(run_dir / "execution.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("Checkpoint run directory: %s", run_dir)

    total = len(subjects)
    logger.info("Running %d subjects", total)

    results: list[dict] = []
    batch_start = time.perf_counter()
    exit_code = 0

    for i, subject_id in enumerate(subjects, start=1):
        # Console progress line (concise).
        elapsed = time.perf_counter() - batch_start
        if i > 1:
            avg_so_far = elapsed / (i - 1)
            eta = avg_so_far * (total - i + 1)
        else:
            eta = 0.0

        result = _execute_subject(
            subject_id=subject_id,
            index=i,
            total=total,
            run_dir=run_dir,
            logger=logger,
            stop_on_error=args.stop_on_error,
        )
        results.append(result)

        # Print concise console progress.
        status_str = result["status"]
        stage_info = f" {result.get('failed_stage', '')}" if status_str == "FAIL" else ""
        print(
            f"[{i:03d}/{total}] {subject_id} {status_str}{stage_info} "
            f"({result['subject_runtime']:.1f} s) "
            f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
        )

        if status_str == "FAIL" and args.stop_on_error:
            logger.error("stop_on_error set: terminating batch after %s failure.", subject_id)
            exit_code = 1
            break

    total_runtime = time.perf_counter() - batch_start

    # Write outputs.
    _write_summary_csv(results, run_dir / "summary.csv")
    _write_failures_csv(results, run_dir / "failures.csv")
    _write_summary_md(results, run_dir, total_runtime, len(discovered), logger)

    logger.info("Checkpoint run complete: %d/%d passed, total runtime %.2fs",
                sum(1 for r in results if r["status"] == "PASS"), len(results), total_runtime)
    logger.info("Outputs in %s", run_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
