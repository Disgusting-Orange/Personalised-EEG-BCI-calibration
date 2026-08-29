"""PyTorch Geometric GNNExplainer wrapper for GCN and GAT Regressors.

Extracts electrode node feature attributions, functional connectivity edge masks,
and spectral band importance breakdowns across subjects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.graph.dataset import EEGGraphDataset
from src.graph.gat_model import GATRegressor
from src.graph.gcn_model import GCNRegressor

logger = logging.getLogger("graph.explainability")

EEG_CHANNELS_64 = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6", "Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz",
    "Af4", "Af8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "Ft7", "Ft8",
    "T7", "T8", "T9", "T10", "Tp7", "Tp8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz"
]

FEATURE_NAMES_10 = [
    "R01_delta", "R01_theta", "R01_alpha", "R01_beta", "R01_gamma",
    "R02_delta", "R02_theta", "R02_alpha", "R02_beta", "R02_gamma"
]


def create_model_instance(model_id: str, in_channels: int = 10) -> torch.nn.Module:
    """Instantiate trained model for explainability."""
    if model_id == "gcn":
        return GCNRegressor(in_channels=in_channels, hidden_channels=64, num_layers=3)
    elif model_id == "gat":
        return GATRegressor(in_channels=in_channels, hidden_channels=32, heads=4, num_layers=3)
    else:
        raise ValueError(f"Unknown model_id '{model_id}'")


def explain_single_graph(
    model: torch.nn.Module,
    data: Data,
    epochs: int = 200,
    lr: float = 0.01,
) -> dict[str, Any]:
    """Generate GNNExplainer feature & edge masks for a single subject graph."""
    model.eval()

    model_config = ModelConfig(
        mode="regression",
        task_level="graph",
        return_type="raw",
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config,
    )

    explanation = explainer(data.x, data.edge_index, edge_weight=data.edge_weight)

    node_mask = explanation.node_mask.detach().cpu().numpy()  # (64, 10)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()  # (E,)

    # Normalize masks to [0, 1]
    if node_mask.max() > node_mask.min():
        node_mask = (node_mask - node_mask.min()) / (node_mask.max() - node_mask.min())

    if edge_mask.max() > edge_mask.min():
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min())

    node_importances = node_mask.sum(axis=1)  # (64,)
    band_importances = node_mask.sum(axis=0)  # (10,)

    return {
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "node_importances": node_importances,
        "band_importances": band_importances,
        "edge_index": data.edge_index.cpu().numpy(),
    }


def explain_cohort(
    model_id: str,
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    subjects: Sequence[str] | None = None,
    epochs: int = 200,
) -> dict[str, Any]:
    """Extract cohort-averaged node, edge, and spectral band attributions across all subjects."""
    out_path = Path(output_dir) / model_id
    out_path.mkdir(parents=True, exist_ok=True)

    ds = EEGGraphDataset(root=dataset_dir)

    if subjects:
        sub_set = set(subjects)
        data_list = [ds[i] for i in range(len(ds)) if getattr(ds[i], "subject_id", f"S{i+1:03d}") in sub_set]
    else:
        data_list = [ds[i] for i in range(len(ds))]

    n_subjects = len(data_list)
    model = create_model_instance(model_id, in_channels=10)

    cohort_node_imp = np.zeros((n_subjects, 64))
    cohort_band_imp = np.zeros((n_subjects, 10))

    # Dense edge importance accumulator (64, 64)
    edge_matrix_acc = np.zeros((64, 64))
    edge_counts = np.zeros((64, 64))

    logger.info("Extracting GNNExplainer masks for model '%s' across %d subjects...", model_id, n_subjects)

    for i, data in enumerate(data_list):
        res = explain_single_graph(model, data, epochs=epochs)

        cohort_node_imp[i] = res["node_importances"]
        cohort_band_imp[i] = res["band_importances"]

        edge_idx = res["edge_index"]
        edge_m = res["edge_mask"]

        for e_idx in range(edge_idx.shape[1]):
            u, v = edge_idx[0, e_idx], edge_idx[1, e_idx]
            edge_matrix_acc[u, v] += edge_m[e_idx]
            edge_counts[u, v] += 1.0

    mean_node_imp = cohort_node_imp.mean(axis=0)
    mean_band_imp = cohort_band_imp.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_edge_matrix = np.where(edge_counts > 0, edge_matrix_acc / edge_counts, 0.0)

    # Save Node Importances CSV
    df_nodes = pd.DataFrame({
        "channel": EEG_CHANNELS_64[:len(mean_node_imp)],
        "importance": mean_node_imp,
    }).sort_values(by="importance", ascending=False)
    df_nodes.to_csv(out_path / "node_importances.csv", index=False)

    # Save Band Importances CSV
    df_bands = pd.DataFrame({
        "feature": FEATURE_NAMES_10[:len(mean_band_imp)],
        "importance": mean_band_imp,
    }).sort_values(by="importance", ascending=False)
    df_bands.to_csv(out_path / "band_importances.csv", index=False)

    # Extract top edges
    edge_rows = []
    u_idx, v_idx = np.where(mean_edge_matrix > 0)
    for u, v in zip(u_idx, v_idx):
        if u < v:  # Undirected pair
            edge_rows.append({
                "source": EEG_CHANNELS_64[u],
                "target": EEG_CHANNELS_64[v],
                "importance": float(mean_edge_matrix[u, v]),
            })

    df_edges = pd.DataFrame(edge_rows).sort_values(by="importance", ascending=False)
    df_edges.to_csv(out_path / "edge_importances.csv", index=False)
    np.save(out_path / "edge_importance_matrix.npy", mean_edge_matrix)

    logger.info("Explainer complete for '%s'. Saved outputs at %s", model_id, out_path)

    return {
        "model_id": model_id,
        "mean_node_importances": mean_node_imp,
        "mean_band_importances": mean_band_imp,
        "mean_edge_matrix": mean_edge_matrix,
        "df_nodes": df_nodes,
        "df_edges": df_edges,
        "df_bands": df_bands,
    }
