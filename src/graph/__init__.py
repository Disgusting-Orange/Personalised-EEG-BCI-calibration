"""Stage 10–14 — PyTorch Geometric Graph Neural Network, Explainability & Statistical Validation Framework.
"""

from src.graph.builder import build_subject_graph
from src.graph.dataset import EEGGraphDataset
from src.graph.explainability import explain_cohort, explain_single_graph
from src.graph.gat_model import GATRegressor
from src.graph.gcn_model import GCNRegressor
from src.graph.statistical_tests import (
    bootstrap_confidence_intervals,
    compute_cohens_dz,
    compute_loso_fold_robustness,
    compute_rank_biserial,
    paired_model_comparisons,
    target_permutation_test,
)

from src.graph.validator import validate_graph, validate_graph_dataset_directory

__all__ = [
    "build_subject_graph",
    "EEGGraphDataset",
    "explain_single_graph",
    "explain_cohort",
    "GCNRegressor",
    "GATRegressor",
    "validate_graph",
    "validate_graph_dataset_directory",
    "paired_model_comparisons",
    "bootstrap_confidence_intervals",
    "target_permutation_test",
    "compute_cohens_dz",
    "compute_rank_biserial",
    "compute_loso_fold_robustness",
]
