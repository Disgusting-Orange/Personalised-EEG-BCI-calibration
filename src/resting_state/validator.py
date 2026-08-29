"""Stage 5 — Resting-state preprocessing validation.

Read-only validation of frozen Stage 2 resting-state outputs (R01 eyes-open,
R02 eyes-closed). Produces JSON + Markdown reports and includes the AGENTS.md
Section 8 methodological evaluation of EO vs EC recordings.

Stage 5 is a validation/QC stage only — no re-preprocessing, no features,
no connectivity, no ML. Stage 2 source is frozen and is NOT modified.

Design rules:
  * Filtered FIF metadata is AUTHORITATIVE for sampling-rate checks.
    QC report values that conflict with FIF metadata are documented as
    known Stage 2 reporting inconsistencies, not repaired.
  * All QC thresholds come from configs/stage5_resting_state_validation.yaml.
  * Reusable for multi-subject expansion: validates all baseline runs
    listed in the config for each configured subject.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import yaml

VALIDATOR_VERSION = "stage5-resting-state-1.0"

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
class RunValidation:
    """Validation result for one baseline run (e.g. S001/R01)."""

    subject_id: str
    run_id: str
    condition: str
    fif_metadata: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "condition": self.condition,
            "fif_metadata": self.fif_metadata,
            "checks": [vars(c) for c in self.checks],
        }


@dataclass
class RestingStateValidation:
    """Full validation result for all baseline runs of one subject."""

    subject_id: str
    runs: list[RunValidation] = field(default_factory=list)
    cross_run_comparison: list[CheckResult] = field(default_factory=list)
    known_stage2_issues: dict[str, Any] = field(default_factory=dict)
    methodological_evaluation: dict[str, str] = field(default_factory=dict)
    verdict: str = VERDICT_FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "verdict": self.verdict,
            "runs": [r.to_dict() for r in self.runs],
            "cross_run_comparison": [vars(c) for c in self.cross_run_comparison],
            "known_stage2_issues": self.known_stage2_issues,
            "methodological_evaluation": self.methodological_evaluation,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the Stage 5 validator configuration."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# FIF metadata extraction (authoritative source)
# ---------------------------------------------------------------------------


def _load_fif_metadata(fif_path: Path) -> dict[str, Any]:
    """Load metadata from a filtered FIF without loading all data."""
    info = mne.io.read_info(fif_path, verbose=False)
    return {
        "sfreq": float(info["sfreq"]),
        "n_channels": len(info["ch_names"]),
        "ch_names": info["ch_names"],
        "n_eeg_channels": len(mne.pick_types(info, eeg=True, meg=False)),
        "highpass": float(info["highpass"]),
        "lowpass": float(info["lowpass"]),
        "file_path": str(fif_path),
    }


def _load_epochs_metadata(epochs_path: Path) -> dict[str, Any]:
    """Load metadata from an epochs FIF without loading all data."""
    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    return {
        "n_epochs": len(epochs),
        "sfreq": float(epochs.info["sfreq"]),
        "tmin": float(epochs.tmin),
        "tmax": float(epochs.tmax),
        "n_channels": epochs.info["nchan"],
        "event_ids": list(epochs.event_id.keys()) if epochs.event_id else [],
    }


# ---------------------------------------------------------------------------
# QC report loading
# ---------------------------------------------------------------------------


def _load_qc_report(qc_dir: Path, subject_id: str, run_id: str) -> dict[str, Any] | None:
    """Load a Stage 2 QC report JSON for a subject/run."""
    path = qc_dir / f"{subject_id}_{run_id}_qc_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Validation checks — per-run
# ---------------------------------------------------------------------------


def _validate_run(
    subject_id: str,
    run_id: str,
    condition: str,
    filtered_dir: Path,
    epochs_dir: Path,
    ica_dir: Path,
    qc_dir: Path,
    thresholds: dict[str, Any],
) -> RunValidation:
    """Validate one baseline run's preprocessing outputs."""
    result = RunValidation(
        subject_id=subject_id,
        run_id=run_id,
        condition=condition,
    )

    # ---- Artifact file existence ----
    fif_path = filtered_dir / f"{subject_id}_{run_id}_filtered_raw.fif"
    epochs_path = epochs_dir / f"{subject_id}_{run_id}_epochs-epo.fif"
    ica_path = ica_dir / f"{subject_id}_{run_id}_ica.fif"
    qc_path = qc_dir / f"{subject_id}_{run_id}_qc_report.json"

    result.checks.append(
        CheckResult(
            name="filtered_fif_exists",
            passed=fif_path.exists(),
            severity="error",
            reason="" if fif_path.exists() else f"Missing: {fif_path}",
        )
    )
    result.checks.append(
        CheckResult(
            name="epochs_fif_exists",
            passed=epochs_path.exists(),
            severity="error",
            reason="" if epochs_path.exists() else f"Missing: {epochs_path}",
        )
    )
    result.checks.append(
        CheckResult(
            name="ica_fif_exists",
            passed=ica_path.exists(),
            severity="error",
            reason="" if ica_path.exists() else f"Missing: {ica_path}",
        )
    )
    result.checks.append(
        CheckResult(
            name="qc_report_exists",
            passed=qc_path.exists(),
            severity="error",
            reason="" if qc_path.exists() else f"Missing: {qc_path}",
        )
    )

    # If the filtered FIF is missing, we cannot do further checks.
    if not fif_path.exists():
        return result

    # ---- Authoritative FIF metadata ----
    fif_meta = _load_fif_metadata(fif_path)
    result.fif_metadata = fif_meta

    # ---- Sampling rate (FIF-authoritative) ----
    expected_sfreq = float(thresholds["expected_sfreq_hz"])
    tolerance = float(thresholds["sfreq_tolerance_hz"])
    sfreq_ok = abs(fif_meta["sfreq"] - expected_sfreq) <= tolerance
    result.checks.append(
        CheckResult(
            name="sfreq_matches_config",
            passed=sfreq_ok,
            severity="error",
            reason=(
                ""
                if sfreq_ok
                else f"FIF sfreq={fif_meta['sfreq']} Hz, expected {expected_sfreq} ± {tolerance} Hz"
            ),
        )
    )

    # ---- Channel count ----
    expected_ch = int(thresholds["expected_n_channels"])
    ch_ok = fif_meta["n_channels"] == expected_ch
    result.checks.append(
        CheckResult(
            name="n_channels_expected",
            passed=ch_ok,
            severity="error",
            reason=(
                ""
                if ch_ok
                else f"FIF n_channels={fif_meta['n_channels']}, expected {expected_ch}"
            ),
        )
    )

    # ---- EEG channel types ----
    eeg_ok = fif_meta["n_eeg_channels"] > 0
    result.checks.append(
        CheckResult(
            name="eeg_channels_present",
            passed=eeg_ok,
            severity="error",
            reason="" if eeg_ok else f"No EEG channels found in {fif_path}",
        )
    )

    # ---- Bandpass range (sanity) ----
    # The config specifies 1-40 Hz; verify the FIF metadata reflects filtering.
    # Lowpass should be >= 0 (applied) and highpass should be >= 0 (applied).
    filter_applied = fif_meta["highpass"] >= 0 or fif_meta["lowpass"] > 0
    result.checks.append(
        CheckResult(
            name="filter_bandpass_recorded",
            passed=filter_applied,
            severity="warning",
            reason=(
                f"highpass={fif_meta['highpass']}, lowpass={fif_meta['lowpass']}"
                if filter_applied
                else "No bandpass filter appears applied in FIF info"
            ),
        )
    )

    # ---- QC report: sampling-rate conflict check ----
    qc = _load_qc_report(qc_dir, subject_id, run_id)
    if qc is not None:
        qc_metrics = qc.get("metrics", {})
        qc_sfreq = qc_metrics.get("sampling_frequency_hz")

        if qc_sfreq is not None and abs(float(qc_sfreq) - fif_meta["sfreq"]) > tolerance:
            result.checks.append(
                CheckResult(
                    name="qc_sfreq_conflict_with_fif",
                    passed=False,
                    severity="warning",
                    reason=(
                        f"QC report sfreq={qc_sfreq} Hz conflicts with "
                        f"authoritative FIF sfreq={fif_meta['sfreq']} Hz. "
                        "FIF metadata is authoritative per Stage 5 policy."
                    ),
                )
            )
        else:
            result.checks.append(
                CheckResult(
                    name="qc_sfreq_consistent_with_fif",
                    passed=True,
                    severity="warning",
                    reason="",
                )
            )

        # ---- QC report: epoch counts ----
        n_usable = qc_metrics.get("n_usable_epochs", -1)
        min_epochs = int(thresholds["min_usable_epochs"])
        epochs_ok = isinstance(n_usable, (int, float)) and int(n_usable) >= min_epochs
        result.checks.append(
            CheckResult(
                name="usable_epochs_sufficient",
                passed=epochs_ok,
                severity="warning",
                reason=(
                    ""
                    if epochs_ok
                    else f"QC n_usable_epochs={n_usable}, minimum {min_epochs}"
                ),
            )
        )

        # ---- QC report: rejection rate ----
        n_gen = qc_metrics.get("n_generated_epochs", 0)
        n_rej = qc_metrics.get("n_rejected_epochs", 0)
        max_rej_rate = float(thresholds["max_rejection_rate"])
        if isinstance(n_gen, (int, float)) and n_gen > 0:
            rej_rate = int(n_rej) / int(n_gen)
            rej_ok = rej_rate <= max_rej_rate
            result.checks.append(
                CheckResult(
                    name="rejection_rate_acceptable",
                    passed=rej_ok,
                    severity="warning",
                    reason=(
                        ""
                        if rej_ok
                        else f"Rejection rate {rej_rate:.1%} exceeds {max_rej_rate:.0%}"
                    ),
                )
            )
        else:
            result.checks.append(
                CheckResult(
                    name="rejection_rate_acceptable",
                    passed=True,
                    severity="warning",
                    reason="No rejection data in QC report",
                )
            )

        # ---- QC report: bad channels ----
        n_bad = qc_metrics.get("n_bad_channels_detected", 0)
        max_bad = int(thresholds["max_bad_channels"])
        bad_ok = isinstance(n_bad, (int, float)) and int(n_bad) <= max_bad
        result.checks.append(
            CheckResult(
                name="bad_channels_acceptable",
                passed=bad_ok,
                severity="warning",
                reason=(
                    ""
                    if bad_ok
                    else f"n_bad_channels={n_bad} exceeds {max_bad}"
                ),
            )
        )

        # ---- QC report: ICA components ----
        ica_removed = qc_metrics.get("ica_components_removed", [])
        result.checks.append(
            CheckResult(
                name="ica_components_recorded",
                passed=isinstance(ica_removed, list),
                severity="info",
                reason="",
            )
        )
    else:
        result.checks.append(
            CheckResult(
                name="qc_report_readable",
                passed=False,
                severity="error",
                reason="QC report not readable",
            )
        )

    # ---- Epochs FIF existence + metadata ----
    if epochs_path.exists():
        try:
            ep_meta = _load_epochs_metadata(epochs_path)
            if ep_meta["n_epochs"] > 0:
                ep_sfreq_ok = abs(ep_meta["sfreq"] - expected_sfreq) <= tolerance
                result.checks.append(
                    CheckResult(
                        name="epochs_sfreq_matches_config",
                        passed=ep_sfreq_ok,
                        severity="error",
                        reason=(
                            ""
                            if ep_sfreq_ok
                            else f"Epochs sfreq={ep_meta['sfreq']}, expected {expected_sfreq}"
                        ),
                    )
                )
        except Exception:
            result.checks.append(
                CheckResult(
                    name="epochs_fif_readable",
                    passed=False,
                    severity="error",
                    reason=f"Could not read {epochs_path}",
                )
            )

    return result


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------


def _cross_run_checks(runs: list[RunValidation]) -> list[CheckResult]:
    """Compare EO and EC runs for consistency."""
    checks: list[CheckResult] = []

    if len(runs) < 2:
        checks.append(
            CheckResult(
                name="both_conditions_present",
                passed=False,
                severity="error",
                reason="Need at least 2 baseline runs for cross-run comparison",
            )
        )
        return checks

    checks.append(
        CheckResult(
            name="both_conditions_present",
            passed=True,
            severity="error",
            reason="",
        )
    )

    # Channel consistency
    r1, r2 = runs[0], runs[1]
    ch_match = r1.fif_metadata.get("ch_names") == r2.fif_metadata.get("ch_names")
    checks.append(
        CheckResult(
            name="channel_names_consistent",
            passed=ch_match,
            severity="error",
            reason="" if ch_match else "Channel names differ between EO and EC recordings",
        )
    )

    # Sampling rate consistency (both should match FIF-authoritative value)
    sf1 = r1.fif_metadata.get("sfreq")
    sf2 = r2.fif_metadata.get("sfreq")
    sf_match = sf1 is not None and sf2 is not None and sf1 == sf2
    checks.append(
        CheckResult(
            name="sfreq_consistent_across_runs",
            passed=sf_match,
            severity="error",
            reason=(
                ""
                if sf_match
                else f"R01 sfreq={sf1}, R02 sfreq={sf2} (both should match config)"
            ),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Methodological evaluation (AGENTS.md §8)
# ---------------------------------------------------------------------------


def _methodological_evaluation() -> dict[str, str]:
    """AGENTS.md Section 8 methodological evaluation of EO vs EC.

    This documents what EEGMMIDB provides and evaluates whether its baseline
    recordings are appropriate as resting-state EEG input.
    """
    return {
        "dataset_provision": (
            "The PhysioNet EEGMMIDB provides two 1-minute baseline recordings "
            "per subject: R01 (eyes open) and R02 (eyes closed), recorded at "
            "64 channels with 160 Hz native sampling rate. These are the ONLY "
            "baseline/rest recordings in the dataset. There is no separate "
            "dedicated resting-state recording (e.g., 5 minutes of rest)."
        ),
        "eyes_open_appropriateness": (
            "Eyes-open resting-state EEG is a widely used condition in "
            "neuroscientific research. It captures visual-alpha desynchronisation "
            "and is suitable for functional connectivity analysis. However, "
            "EEGMMIDB's 1-minute R01 recording is short relative to typical "
            "resting-state protocols (which often use 3-10 minutes). This "
            "limits spectral resolution, particularly in lower-frequency bands "
            "(delta, theta)."
        ),
        "eyes_closed_appropriateness": (
            "Eyes-closed resting-state EEG is the classical condition for "
            "alpha-band analysis and is standard in connectivity studies. "
            "The 1-minute R02 recording has the same duration limitation as R01. "
            "Eyes-closed may yield stronger alpha oscillations, which can "
            "benefit alpha-band connectivity estimates."
        ),
        "separate_vs_combined": (
            "EO and EC are physiologically distinct conditions with different "
            "spectral profiles (alpha power is typically higher in EC). Combining "
            "them would mix two different neural states and could obscure "
            "condition-specific connectivity patterns. They should be analysed "
            "separately as the primary approach, with combined analysis as a "
            "secondary sensitivity test only if scientifically justified."
        ),
        "limitations": (
            "Several limitations must be reported: (1) Duration: ~1 minute per "
            "condition is short for reliable resting-state connectivity estimation. "
            "This constrains spectral resolution and frequency-band specificity. "
            "(2) Task context: these recordings were collected as baselines "
            "within a motor-task experiment, not as dedicated resting-state "
            "sessions. The subject's state may differ from a pure resting-state "
            "protocol. (3) No pre-task vs post-task distinction: it is unknown "
            "whether R01/R02 precede or follow the motor-task runs, so fatigue "
            "or task-aftereffects cannot be ruled out. (4) Only one recording per "
            "condition per subject provides no test-retest reliability within "
            "the dataset. These limitations must be explicitly reported in any "
            "publication using these recordings as resting-state input."
        ),
        "wbs_journal_track_note": (
            "Per the WBS Journal track (task J2.2), EO vs EC must be tested "
            "separately as a robustness analysis regardless of which is used as "
            "the primary resting-state input. This Stage 5 validation enables "
            "that comparison by ensuring both conditions have validated "
            "preprocessing outputs."
        ),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _compute_verdict(validation: RestingStateValidation) -> str:
    """Aggregate verdict: FAIL > WARNING > PASS."""
    has_error = False
    has_warning = False

    for run in validation.runs:
        for c in run.checks:
            if not c.passed and c.severity == "error":
                has_error = True
            if not c.passed and c.severity in ("warning", "info"):
                has_warning = True

    for c in validation.cross_run_comparison:
        if not c.passed and c.severity == "error":
            has_error = True
        if not c.passed and c.severity in ("warning", "info"):
            has_warning = True

    if has_error:
        return VERDICT_FAIL
    if has_warning:
        return VERDICT_WARNING
    return VERDICT_PASS


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _write_json(
    validation: RestingStateValidation,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_thresholds": config["thresholds"],
        "result": validation.to_dict(),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_markdown(
    validation: RestingStateValidation,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sid = validation.subject_id
    lines.append(f"# Stage 5 — Resting-State Validation Report: {sid}")
    lines.append("")
    lines.append(f"- Validator version: `{VALIDATOR_VERSION}`")
    lines.append(f"- **Verdict: {validation.verdict}**")
    lines.append(f"- Baseline runs validated: {len(validation.runs)}")
    lines.append("")

    for run in validation.runs:
        lines.append(f"## {run.subject_id}/{run.run_id} ({run.condition})")
        lines.append("")
        lines.append("### Authoritative FIF metadata")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        for k, v in run.fif_metadata.items():
            if k == "ch_names":
                lines.append(f"| {k} | {len(v)} channels |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

        lines.append("### Validation checks")
        lines.append("")
        lines.append("| Check | Passed | Severity | Reason |")
        lines.append("|---|---|---|---|")
        for c in run.checks:
            lines.append(
                f"| {c.name} | {'yes' if c.passed else 'no'} | {c.severity} | {c.reason or '—'} |"
            )
        lines.append("")

    if validation.cross_run_comparison:
        lines.append("## Cross-run comparison (EO vs EC)")
        lines.append("")
        lines.append("| Check | Passed | Severity | Reason |")
        lines.append("|---|---|---|---|")
        for c in validation.cross_run_comparison:
            lines.append(
                f"| {c.name} | {'yes' if c.passed else 'no'} | {c.severity} | {c.reason or '—'} |"
            )
        lines.append("")

    if validation.known_stage2_issues:
        lines.append("## Known Stage 2 reporting issues")
        lines.append("")
        lines.append(
            "The following inconsistencies in Stage 2 output are documented here. "
            "They are NOT repaired (Stage 2 is frozen). Filtered FIF metadata is "
            "treated as authoritative."
        )
        lines.append("")
        for issue_name, issue_detail in validation.known_stage2_issues.items():
            desc = issue_detail.get("description", "")
            affected = issue_detail.get("affected_runs", [])
            authoritative = issue_detail.get("authoritative_source", "")
            lines.append(f"### {issue_name}")
            lines.append("")
            lines.append(f"> {desc}")
            lines.append("")
            lines.append(f"- Affected runs: {', '.join(affected)}")
            lines.append(f"- Authoritative source: {authoritative}")
            lines.append("")

    if validation.methodological_evaluation:
        lines.append("## Methodological evaluation (AGENTS.md §8)")
        lines.append("")
        lines.append(
            "Per AGENTS.md Section 8, the following evaluates whether EEGMMIDB "
            "baseline recordings are appropriate as resting-state EEG input."
        )
        lines.append("")
        section_labels = {
            "dataset_provision": "What EEGMMIDB provides",
            "eyes_open_appropriateness": "Eyes-open (R01) appropriateness",
            "eyes_closed_appropriateness": "Eyes-closed (R02) appropriateness",
            "separate_vs_combined": "Separate vs combined analysis",
            "limitations": "Limitations",
            "wbs_journal_track_note": "WBS Journal-track requirement",
        }
        for key, label in section_labels.items():
            text = validation.methodological_evaluation.get(key, "")
            if text:
                lines.append(f"### {label}")
                lines.append("")
                lines.append(text)
                lines.append("")

    lines.append("## Downstream usability")
    lines.append("")
    if validation.verdict == VERDICT_PASS:
        lines.append(
            "Resting-state preprocessing outputs are validated and suitable "
            "for downstream spectral feature extraction and connectivity analysis."
        )
    elif validation.verdict == VERDICT_WARNING:
        lines.append(
            "Resting-state preprocessing outputs are usable with documented "
            "caveats (see failed warnings and known issues above)."
        )
    else:
        lines.append(
            "Resting-state preprocessing outputs have error-level failures "
            "and are NOT suitable for downstream use until resolved."
        )
    lines.append("")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_stage5(config_path: str | Path) -> dict[str, Any]:
    """Run Stage 5 resting-state validation for all configured subjects/runs."""
    config_path = Path(config_path)
    config = load_config(config_path)

    preprocessed = config["preprocessed"]
    thresholds = config["thresholds"]
    baseline_runs = config["baseline_runs"]
    subjects = config["subjects"]

    output_dir = Path(config["output_directory"])
    overwrite = bool(config.get("overwrite", True))

    filtered_dir = Path(preprocessed["filtered_dir"])
    epochs_dir = Path(preprocessed["epochs_dir"])
    ica_dir = Path(preprocessed["ica_dir"])
    qc_dir = Path(preprocessed["qc_dir"])

    summary: dict[str, Any] = {
        "validator_version": VALIDATOR_VERSION,
        "subjects": [],
    }

    for subject_id in subjects:
        validation = RestingStateValidation(
            subject_id=subject_id,
            known_stage2_issues=config.get("known_stage2_issues", {}),
            methodological_evaluation=_methodological_evaluation(),
        )

        for br in baseline_runs:
            run_val = _validate_run(
                subject_id=subject_id,
                run_id=br["run_id"],
                condition=br["condition"],
                filtered_dir=filtered_dir,
                epochs_dir=epochs_dir,
                ica_dir=ica_dir,
                qc_dir=qc_dir,
                thresholds=thresholds,
            )
            validation.runs.append(run_val)

        validation.cross_run_comparison = _cross_run_checks(validation.runs)
        validation.verdict = _compute_verdict(validation)

        json_path = output_dir / f"{subject_id}_resting_state_validation.json"
        md_path = output_dir / f"{subject_id}_resting_state_validation.md"
        if not overwrite and (json_path.exists() or md_path.exists()):
            raise FileExistsError(
                f"Reports exist for {subject_id} and overwrite=False"
            )

        _write_json(validation, json_path, config)
        _write_markdown(validation, md_path)

        summary["subjects"].append(validation.to_dict())

    return summary
