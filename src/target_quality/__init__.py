"""Stage 4 — Target quality and reliability validation.

This package validates Stage 3 target-generation outputs to determine
whether the continuous MI decoding target is traceable, internally
consistent, and reliable enough for downstream modelling.

Stage 4 is validation / quality-control only — NO statistical inference.
"""

from .validator import (
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_WARNING,
    TargetValidation,
    discover_subjects,
    load_config,
    run_stage4,
    validate_subject_target,
    write_reports,
)

__all__ = [
    "VERDICT_FAIL",
    "VERDICT_PASS",
    "VERDICT_WARNING",
    "TargetValidation",
    "discover_subjects",
    "load_config",
    "run_stage4",
    "validate_subject_target",
    "write_reports",
]
