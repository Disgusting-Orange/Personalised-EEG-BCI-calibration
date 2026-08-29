"""Stage 8 — Classical Baseline Regressors Module.

Provides a publication-quality baseline benchmark suite for continuous
motor imagery decoding performance prediction using spectral and functional
connectivity features under leakage-free Leave-One-Subject-Out (LOSO) nested CV.
"""

from src.baselines.evaluator import evaluate_model_loso, load_or_create_loso_splits
from src.baselines.feature_loader import load_dataset
from src.baselines.models import create_pipeline, get_hpo_grid
from src.baselines.stats import compute_bootstrap_cis, compute_regression_metrics, paired_model_comparison
from src.baselines.visualization import (
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
)

__all__ = [
    "load_dataset",
    "create_pipeline",
    "get_hpo_grid",
    "load_or_create_loso_splits",
    "evaluate_model_loso",
    "compute_regression_metrics",
    "compute_bootstrap_cis",
    "paired_model_comparison",
    "plot_predicted_vs_actual",
    "plot_residuals",
    "plot_feature_importance",
    "plot_model_comparison",
]
