"""Stage 4 — Target quality and reliability validation.

This module validates that a Stage 3 target-generation output directory
produces a traceable, internally consistent, and reliability-quantified
continuous MI decoding target that is suitable for downstream modelling.

Stage 4 is a validation / quality-control stage only. It performs NO
statistical inference: no bootstrap confidence intervals, no permutation
tests, no significance tests.

Design rules (AGENTS.md + project working rules):
  * Validates SCHEMA, not expected constant values. The validator never
    fails because a subject ID, decoder parameter, or CV scheme differs
    from a hardcoded expectation — only because a required field is
    missing or internally inconsistent.
  * All QC thresholds live in configs/stage4_target_quality.yaml.
    None are hardcoded here.
  * Reusable for future multi-subject target directories: the validator
    operates on a target directory path and discovers subjects from the
    Stage 3 outputs, rather than being wired to S001.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VALIDATOR_VERSION = "stage4-target-quality-1.0"

VERDICT_PASS = "PASS"
VERDICT_WARNING = "WARNING"
VERDICT_FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    severity: str  # "error" or "warning"
    reason: str = ""


@dataclass
class FlagResult:
    """Outcome of a single QC flag (uses configured thresholds)."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    reason: str = ""


@dataclass
class TargetValidation:
    """Full validation result for one subject's target."""

    subject_id: str
    traceability: list[CheckResult] = field(default_factory=list)
    integrity: list[CheckResult] = field(default_factory=list)
    schema_compatibility: list[CheckResult] = field(default_factory=list)
    reliability: dict[str, float] = field(default_factory=dict)
    flags: list[FlagResult] = field(default_factory=list)
    verdict: str = VERDICT_FAIL
    target_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "target_value": self.target_value,
            "verdict": self.verdict,
            "traceability": [vars(c) for c in self.traceability],
            "integrity": [vars(c) for c in self.integrity],
            "schema_compatibility": [vars(c) for c in self.schema_compatibility],
            "reliability": self.reliability,
            "flags": [vars(f) for f in self.flags],
        }


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the Stage 4 validator configuration."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Stage 3 output discovery
# ---------------------------------------------------------------------------


def _find_target_report(target_dir: Path, subject_id: str) -> Path | None:
    candidates = sorted(target_dir.glob(f"{subject_id}_target_report.json"))
    return candidates[0] if candidates else None


def _find_fold_metrics(target_dir: Path, subject_id: str) -> Path | None:
    candidates = sorted(target_dir.glob(f"{subject_id}_fold_metrics.csv"))
    return candidates[0] if candidates else None


def _find_mi_targets(target_dir: Path) -> Path | None:
    candidates = sorted(target_dir.glob("mi_targets.csv"))
    return candidates[0] if candidates else None


def discover_subjects(target_dir: Path) -> list[str]:
    """Discover subject IDs present in a Stage 3 target directory.

    Robust to single-subject (S001 only) and future multi-subject layouts:
    subjects are inferred from `<subject_id>_target_report.json` filenames.
    """
    subjects: list[str] = []
    for path in sorted(target_dir.glob("*_target_report.json")):
        stem = path.stem
        if stem.endswith("_target_report"):
            subjects.append(stem[: -len("_target_report")])
    return subjects


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _check_traceability(
    report: dict[str, Any],
    fold_path: Path | None,
    mi_targets_path: Path | None,
    schema: dict[str, list[str]],
) -> list[CheckResult]:
    """Traceability validation: is the target's origin fully recorded?"""
    results: list[CheckResult] = []

    # subject_id present
    sid = report.get("subject_id")
    results.append(
        CheckResult(
            name="subject_id_recorded",
            passed=bool(sid),
            severity="error",
            reason="" if sid else "subject_id missing from target_report.json",
        )
    )

    # CV scheme recorded
    cv = report.get("cross_validation")
    cv_ok = isinstance(cv, dict) and all(cv.get(f) is not None for f in schema["cross_validation_required_fields"])
    results.append(
        CheckResult(
            name="cross_validation_recorded",
            passed=cv_ok,
            severity="error",
            reason="" if cv_ok else "cross_validation block missing required fields",
        )
    )

    # Decoder recorded (presence only; values are NOT pinned)
    decoder = report.get("decoder")
    results.append(
        CheckResult(
            name="decoder_recorded",
            passed=isinstance(decoder, dict) and len(decoder) > 0,
            severity="error",
            reason="" if decoder else "decoder block missing",
        )
    )

    # Primary target metric recorded
    metric_name = report.get("primary_target_metric")
    results.append(
        CheckResult(
            name="primary_target_metric_recorded",
            passed=bool(metric_name),
            severity="error",
            reason="" if metric_name else "primary_target_metric not recorded",
        )
    )

    # OOF origin: Stage 3 must record a pre-CV policy indicating the
    # target originates from held-out predictions, not training performance.
    policy = None
    for run in report.get("preprocessing_provenance", {}).get("runs", []):
        policy = run.get("pre_cv_policy")
        if policy:
            break
    oof_marker_present = bool(policy) and "within CV folds" in str(policy)
    results.append(
        CheckResult(
            name="oof_origin_recorded",
            passed=oof_marker_present,
            severity="error",
            reason=(
                ""
                if oof_marker_present
                else "Could not confirm CSP/LDA were fit within CV folds (no pre_cv_policy marker)"
            ),
        )
    )

    # Companion files exist for traceability
    results.append(
        CheckResult(
            name="fold_metrics_file_present",
            passed=fold_path is not None and fold_path.exists(),
            severity="error",
            reason="" if fold_path and fold_path.exists() else "fold metrics CSV missing",
        )
    )
    results.append(
        CheckResult(
            name="mi_targets_file_present",
            passed=mi_targets_path is not None and mi_targets_path.exists(),
            severity="error",
            reason="" if mi_targets_path and mi_targets_path.exists() else "mi_targets.csv missing",
        )
    )

    # Software versions recorded (reproducibility traceability)
    versions = report.get("software_versions")
    results.append(
        CheckResult(
            name="software_versions_recorded",
            passed=isinstance(versions, dict) and len(versions) > 0,
            severity="warning",
            reason="" if versions else "software_versions block missing",
        )
    )

    return results


def _check_schema_compatibility(
    report: dict[str, Any],
    fold_columns: list[str],
    mi_targets_columns: list[str],
    schema: dict[str, list[str]],
) -> list[CheckResult]:
    """Schema compatibility validation for downstream stages.

    Confirms every field that downstream resting-state/feature/regression
    stages will read is actually present. This decouples Stage 4 from any
    specific Stage 3 parameter values: only the field *names* are pinned.
    """
    results: list[CheckResult] = []

    # Top-level fields
    missing_top = [f for f in schema["target_report_required_top_level"] if f not in report]
    results.append(
        CheckResult(
            name="target_report_top_level_fields",
            passed=not missing_top,
            severity="error",
            reason="missing: " + ", ".join(missing_top) if missing_top else "",
        )
    )

    # CV sub-fields
    cv = report.get("cross_validation", {}) or {}
    missing_cv = [f for f in schema["cross_validation_required_fields"] if f not in cv]
    results.append(
        CheckResult(
            name="cross_validation_fields",
            passed=not missing_cv,
            severity="error",
            reason="missing: " + ", ".join(missing_cv) if missing_cv else "",
        )
    )

    # Metrics sub-fields
    metrics = report.get("metrics", {}) or {}
    missing_metrics = [f for f in schema["metrics_required_fields"] if f not in metrics]
    results.append(
        CheckResult(
            name="metrics_fields",
            passed=not missing_metrics,
            severity="error",
            reason="missing: " + ", ".join(missing_metrics) if missing_metrics else "",
        )
    )

    # Fold metrics columns
    missing_fold_cols = [c for c in schema["fold_metrics_required_columns"] if c not in fold_columns]
    results.append(
        CheckResult(
            name="fold_metrics_columns",
            passed=not missing_fold_cols,
            severity="error",
            reason="missing columns: " + ", ".join(missing_fold_cols) if missing_fold_cols else "",
        )
    )

    # mi_targets columns
    missing_mt_cols = [c for c in schema["mi_targets_required_columns"] if c not in mi_targets_columns]
    results.append(
        CheckResult(
            name="mi_targets_columns",
            passed=not missing_mt_cols,
            severity="error",
            reason="missing columns: " + ", ".join(missing_mt_cols) if missing_mt_cols else "",
        )
    )

    return results


def _check_integrity(
    report: dict[str, Any],
    fold_rows: list[dict[str, str]],
    mi_targets_rows: list[dict[str, str]],
    subject_id: str,
) -> list[CheckResult]:
    """Internal consistency checks (numeric finiteness, count coherence)."""
    results: list[CheckResult] = []

    # Primary metric value finite
    metric_name = report.get("primary_target_metric")
    metrics = report.get("metrics", {}) or {}
    value = metrics.get(metric_name) if metric_name else None
    results.append(
        CheckResult(
            name="target_value_finite",
            passed=_is_finite_number(value),
            severity="error",
            reason="" if _is_finite_number(value) else f"{metric_name} is missing or non-finite",
        )
    )

    # Trial count coherence: total == sum(per-class)
    n_total = report.get("n_trials_total")
    n_per_class = report.get("n_trials_per_class", {}) or {}
    try:
        summed = sum(int(v) for v in n_per_class.values())
        coherent = isinstance(n_total, int) and summed == n_total
    except (TypeError, ValueError):
        coherent = False
    results.append(
        CheckResult(
            name="trial_count_coherent",
            passed=coherent,
            severity="error",
            reason=(
                ""
                if coherent
                else f"n_trials_total={n_total} != sum(n_trials_per_class)={n_per_class}"
            ),
        )
    )

    # Per-class counts are positive
    try:
        positive = all(int(v) > 0 for v in n_per_class.values())
    except (TypeError, ValueError):
        positive = False
    results.append(
        CheckResult(
            name="per_class_counts_positive",
            passed=positive,
            severity="error",
            reason="" if positive else "one or more per-class trial counts is non-positive",
        )
    )

    # Fold count matches CV n_splits
    cv = report.get("cross_validation", {}) or {}
    n_splits = cv.get("n_splits")
    n_fold_rows = len(fold_rows)
    folds_match = isinstance(n_splits, int) and n_fold_rows == n_splits
    results.append(
        CheckResult(
            name="fold_count_matches_n_splits",
            passed=folds_match,
            severity="error",
            reason=(
                ""
                if folds_match
                else f"n_splits={n_splits} but fold_metrics has {n_fold_rows} rows"
            ),
        )
    )

    # Fold balanced_accuracy values are finite
    fold_vals = [r.get("balanced_accuracy") for r in fold_rows]
    all_finite = all(_is_finite_number(float(v)) for v in fold_vals if v is not None) and len(fold_vals) > 0
    results.append(
        CheckResult(
            name="fold_balanced_accuracy_finite",
            passed=all_finite,
            severity="error",
            reason="" if all_finite else "one or more fold balanced_accuracy values are missing/non-finite",
        )
    )

    # mi_targets row matches subject and target value
    sid_match = any(r.get("subject_id") == subject_id for r in mi_targets_rows)
    results.append(
        CheckResult(
            name="mi_targets_subject_present",
            passed=sid_match,
            severity="error",
            reason="" if sid_match else f"subject {subject_id} missing from mi_targets.csv",
        )
    )

    return results


def _compute_reliability(fold_scores: list[float]) -> dict[str, float]:
    """Reliability metrics only — no CI, no test, no SD-as-CI mislabelling."""
    return {
        "n_folds": float(len(fold_scores)),
        "mean": float(statistics.fmean(fold_scores)),
        "sd": float(statistics.pstdev(fold_scores)) if len(fold_scores) > 1 else 0.0,
        "min": float(min(fold_scores)),
        "max": float(max(fold_scores)),
    }


def _assess_flags(
    report: dict[str, Any],
    reliability: dict[str, float],
    thresholds: dict[str, Any],
) -> list[FlagResult]:
    """QC flags driven entirely by configured thresholds."""
    flags: list[FlagResult] = []
    metric_name = report.get("primary_target_metric")
    mean_val = reliability.get("mean")
    n_total = report.get("n_trials_total")
    n_per_class = report.get("n_trials_per_class", {}) or {}

    # Above chance (boolean QC, not a test)
    chance = float(thresholds["balanced_accuracy_chance_level"])
    margin = float(thresholds["above_chance_margin"])
    above = isinstance(mean_val, (int, float)) and mean_val > (chance + margin)
    flags.append(
        FlagResult(
            name="above_chance",
            passed=above,
            value=mean_val,
            threshold=f">{chance + margin}",
            reason="" if above else f"mean {metric_name} {mean_val} not above chance {chance}+{margin}",
        )
    )

    # Fold SD within threshold
    sd = reliability.get("sd", 0.0)
    max_sd = float(thresholds["max_fold_sd"])
    sd_ok = sd <= max_sd
    flags.append(
        FlagResult(
            name="fold_sd_acceptable",
            passed=sd_ok,
            value=sd,
            threshold=f"<={max_sd}",
            reason="" if sd_ok else f"fold SD {sd:.4f} exceeds {max_sd}",
        )
    )

    # Sufficient trials
    try:
        n = int(n_total)
    except (TypeError, ValueError):
        n = -1
    min_trials = int(thresholds["min_total_trials"])
    trials_ok = n >= min_trials
    flags.append(
        FlagResult(
            name="sufficient_trials",
            passed=trials_ok,
            value=n,
            threshold=f">={min_trials}",
            reason="" if trials_ok else f"n_trials_total {n} below {min_trials}",
        )
    )

    # Class balance ratio
    try:
        counts = [int(v) for v in n_per_class.values()]
        ratio = max(counts) / min(counts) if min(counts) > 0 else math.inf
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = math.inf
    max_ratio = float(thresholds["max_class_imbalance_ratio"])
    balance_ok = ratio <= max_ratio
    flags.append(
        FlagResult(
            name="class_balance_acceptable",
            passed=balance_ok,
            value=ratio,
            threshold=f"<={max_ratio}",
            reason="" if balance_ok else f"class ratio {ratio:.2f} exceeds {max_ratio}",
        )
    )

    return flags


def _compute_verdict(validation: TargetValidation) -> str:
    """Aggregate verdict: FAIL > WARNING > PASS.

    FAIL = any error-severity check failed.
    WARNING = no errors, but a warning-severity check or a QC flag failed.
    PASS = everything clean.
    """
    has_error = any(
        (not c.passed) and c.severity == "error"
        for c in validation.traceability + validation.integrity + validation.schema_compatibility
    )
    if has_error:
        return VERDICT_FAIL

    has_warning = any(
        (not c.passed) and c.severity == "warning"
        for c in validation.traceability + validation.integrity + validation.schema_compatibility
    )
    has_flag_fail = any(not f.passed for f in validation.flags)
    if has_warning or has_flag_fail:
        return VERDICT_WARNING

    return VERDICT_PASS


# ---------------------------------------------------------------------------
# Per-subject validation entry point
# ---------------------------------------------------------------------------


def validate_subject_target(
    target_dir: Path,
    subject_id: str,
    config: dict[str, Any],
) -> TargetValidation:
    """Validate one subject's Stage 3 target output."""
    schema = config["schema"]
    thresholds = config["thresholds"]

    report_path = _find_target_report(target_dir, subject_id)
    fold_path = _find_fold_metrics(target_dir, subject_id)
    mi_targets_path = _find_mi_targets(target_dir)

    validation = TargetValidation(subject_id=subject_id)

    # If the target report itself is missing, we cannot proceed — short-circuit.
    if report_path is None or not report_path.exists():
        validation.traceability.append(
            CheckResult(
                name="target_report_file_present",
                passed=False,
                severity="error",
                reason=f"{subject_id}_target_report.json not found in {target_dir}",
            )
        )
        validation.verdict = VERDICT_FAIL
        return validation

    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    fold_rows = _read_csv_rows(fold_path) if fold_path and fold_path.exists() else []
    mi_targets_rows = _read_csv_rows(mi_targets_path) if mi_targets_path and mi_targets_path.exists() else []
    fold_columns = list(fold_rows[0].keys()) if fold_rows else []
    mi_targets_columns = list(mi_targets_rows[0].keys()) if mi_targets_rows else []

    validation.traceability = _check_traceability(report, fold_path, mi_targets_path, schema)
    validation.schema_compatibility = _check_schema_compatibility(
        report, fold_columns, mi_targets_columns, schema
    )
    validation.integrity = _check_integrity(report, fold_rows, mi_targets_rows, subject_id)

    # Reliability + flags only meaningful if folds loaded cleanly
    fold_scores = []
    for row in fold_rows:
        try:
            fold_scores.append(float(row["balanced_accuracy"]))
        except (KeyError, TypeError, ValueError):
            pass
    if fold_scores:
        validation.reliability = _compute_reliability(fold_scores)
        validation.flags = _assess_flags(report, validation.reliability, thresholds)

    metric_name = report.get("primary_target_metric")
    metrics = report.get("metrics", {}) or {}
    if metric_name and _is_finite_number(metrics.get(metric_name)):
        validation.target_value = float(metrics[metric_name])

    validation.verdict = _compute_verdict(validation)
    return validation


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_reports(
    validation: TargetValidation,
    target_dir: Path,
    report_path_json: Path,
    report_path_md: Path,
    config: dict[str, Any],
) -> None:
    """Write machine-readable JSON and human-readable Markdown reports."""
    report_path_json.parent.mkdir(parents=True, exist_ok=True)
    report_path_md.parent.mkdir(parents=True, exist_ok=True)

    # ---- JSON ----
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_thresholds": config["thresholds"],
        "target_directory": str(target_dir),
        "result": validation.to_dict(),
    }
    with report_path_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # ---- Markdown ----
    lines: list[str] = []
    lines.append(f"# Stage 4 — Target Quality Report: {validation.subject_id}")
    lines.append("")
    lines.append(f"- Validator version: `{VALIDATOR_VERSION}`")
    lines.append(f"- Target directory: `{target_dir}`")
    lines.append(f"- Primary target value: `{validation.target_value}`")
    lines.append(f"- **Verdict: {validation.verdict}**")
    lines.append("")

    def _section(title: str, checks: list[CheckResult]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Check | Passed | Severity | Reason |")
        lines.append("|---|---|---|---|")
        for c in checks:
            lines.append(
                f"| {c.name} | {'yes' if c.passed else 'no'} | {c.severity} | {c.reason or '—'} |"
            )
        lines.append("")

    _section("Traceability", validation.traceability)
    _section("Schema compatibility (downstream stages)", validation.schema_compatibility)
    _section("Integrity", validation.integrity)

    lines.append("## Reliability metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in validation.reliability.items():
        lines.append(f"| {k} | {v:.6f} |")
    lines.append("")

    lines.append("## Quality flags")
    lines.append("")
    lines.append("| Flag | Passed | Value | Threshold | Reason |")
    lines.append("|---|---|---|---|---|")
    for f in validation.flags:
        val_str = f"{f.value:.4f}" if isinstance(f.value, float) else str(f.value)
        lines.append(
            f"| {f.name} | {'yes' if f.passed else 'no'} | {val_str} | {f.threshold} | {f.reason or '—'} |"
        )
    lines.append("")

    lines.append("## Downstream usability")
    lines.append("")
    if validation.verdict == VERDICT_PASS:
        lines.append("Target is suitable for downstream modelling.")
    elif validation.verdict == VERDICT_WARNING:
        lines.append(
            "Target may be used for downstream modelling, but with documented caveats "
            "(see failed flags / warnings above)."
        )
    else:
        lines.append(
            "Target is NOT suitable for downstream modelling — error-severity checks failed."
        )
    lines.append("")

    with report_path_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Directory-level orchestration
# ---------------------------------------------------------------------------


def run_stage4(config_path: str | Path) -> dict[str, Any]:
    """Run Stage 4 over every subject discovered in the target directory."""
    config_path = Path(config_path)
    config = load_config(config_path)

    target_dir = Path(config["target_directory"])
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    output_dir = Path(config["output_directory"])
    overwrite = bool(config.get("overwrite", True))

    subjects = discover_subjects(target_dir)
    if not subjects:
        raise RuntimeError(f"No subjects discovered in {target_dir}")

    summary: dict[str, Any] = {
        "validator_version": VALIDATOR_VERSION,
        "target_directory": str(target_dir),
        "subjects": [],
    }

    for subject_id in subjects:
        validation = validate_subject_target(target_dir, subject_id, config)

        json_path = output_dir / f"{subject_id}_target_quality.json"
        md_path = output_dir / f"{subject_id}_target_quality.md"
        if not overwrite and (json_path.exists() or md_path.exists()):
            raise FileExistsError(
                f"Reports exist for {subject_id} and overwrite=False: {json_path}, {md_path}"
            )

        write_reports(validation, target_dir, json_path, md_path, config)
        summary["subjects"].append(validation.to_dict())

    return summary
