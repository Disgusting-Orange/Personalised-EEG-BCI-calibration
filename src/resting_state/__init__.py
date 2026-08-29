"""Stage 5 — Resting-state preprocessing validation.

Validates Stage 2 resting-state outputs (R01 eyes-open, R02 eyes-closed)
against expected schemas, configured thresholds, and AGENTS.md requirements.
Produces JSON + Markdown reports and documents the §8 methodological
evaluation of EO vs EC recordings.
"""

from .connectivity import (
    CONNECTIVITY_ESTIMATORS,
    BaseConnectivityEstimator,
    CoherenceEstimator,
    PLVIEstimator,
    WPLIEstimator,
    generate_connectivity_heatmap,
    run_stage7_subject,
)
from .spectral import (
    compute_epoch_psd,
    export_node_features,
    extract_band_powers,
    run_stage6_subject,
)
from .validator import (
    VALIDATOR_VERSION,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_WARNING,
    CheckResult,
    RestingStateValidation,
    run_stage5,
)

__all__ = [
    "VALIDATOR_VERSION",
    "VERDICT_FAIL",
    "VERDICT_PASS",
    "VERDICT_WARNING",
    "CheckResult",
    "RestingStateValidation",
    "run_stage5",
    "compute_epoch_psd",
    "extract_band_powers",
    "export_node_features",
    "run_stage6_subject",
    "BaseConnectivityEstimator",
    "WPLIEstimator",
    "PLVIEstimator",
    "CoherenceEstimator",
    "CONNECTIVITY_ESTIMATORS",
    "generate_connectivity_heatmap",
    "run_stage7_subject",
]
