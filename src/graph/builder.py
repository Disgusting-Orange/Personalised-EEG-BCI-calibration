"""Single-subject PyTorch Geometric Data object builder and edge sparsifier.

Loads frozen outputs from Stage 3 (MI target), Stage 6 (Spectral node features),
and Stage 7 (wPLI connectivity), constructing PyG Data objects with COO edge indices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ABSOLUTE_BAND_COLS = ["delta_abs", "theta_abs", "alpha_abs", "beta_abs", "gamma_abs"]
RELATIVE_BAND_COLS = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"]


def get_standard_3d_coordinates(ch_names: list[str]) -> np.ndarray:
    """Get standardized 3D electrode positions (x, y, z) for 64 EEG channels."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    coords = []
    for ch in ch_names:
        ch_clean = ch.rstrip(".")
        if ch_clean in pos_dict:
            coords.append(pos_dict[ch_clean])
        elif ch in pos_dict:
            coords.append(pos_dict[ch])
        else:
            coords.append([0.0, 0.0, 0.0])
    coords = np.array(coords, dtype=np.float32)
    mu = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    std[std < 1e-6] = 1.0
    return (coords - mu) / std


def build_subject_graph(
    subject_id: str,
    config: dict[str, Any] | None = None,
    targets_dir: Union[str, Path] | None = None,
    features_dir: Union[str, Path] | None = None,
    connectivity_dir: Union[str, Path] | None = None,
) -> Data:
    """Build PyTorch Geometric Data graph object for a single subject."""
    sub_lower = subject_id.lower()
    targets_root = Path(targets_dir) if targets_dir else REPOSITORY_ROOT / "outputs" / "targets"
    features_root = Path(features_dir) if features_dir else REPOSITORY_ROOT / "outputs" / "features"
    conn_root = Path(connectivity_dir) if connectivity_dir else REPOSITORY_ROOT / "outputs" / "connectivity"

    conn_metric = config.get("connectivity_metric", "wpli") if config else "wpli"
    freq_band = config.get("frequency_band", "alpha") if config else "alpha"
    density = float(config.get("sparsification_density", 0.15)) if config else 0.15
    feat_type = config.get("node_feature_type", "concatenated") if config else "concatenated"
    ensure_connected = bool(config.get("ensure_connected", True)) if config else True

    # 1. Load Target (Stage 3)
    target_file = targets_root / f"stage3_{sub_lower}" / "mi_targets.csv"
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found for {subject_id}: {target_file}")

    df_target = pd.read_csv(target_file)
    y_val = float(df_target["balanced_accuracy"].iloc[0])

    # 2. Load Node Features (Stage 6)
    eo_pow_file = features_root / f"stage6_{sub_lower}" / f"{subject_id}_R01_band_powers.csv"
    ec_pow_file = features_root / f"stage6_{sub_lower}" / f"{subject_id}_R02_band_powers.csv"

    if not eo_pow_file.exists() or not ec_pow_file.exists():
        raise FileNotFoundError(f"Spectral features not found for {subject_id}: {eo_pow_file}")

    df_eo = pd.read_csv(eo_pow_file, index_col=0)
    df_ec = pd.read_csv(ec_pow_file, index_col=0)
    ch_names = list(df_eo.index)

    eo_rel = df_eo[RELATIVE_BAND_COLS].values  # (64, 5)
    ec_rel = df_ec[RELATIVE_BAND_COLS].values  # (64, 5)

    if feat_type == "concatenated":
        X_mat = np.hstack([eo_rel, ec_rel])  # (64, 10)
    elif feat_type in ("full_spectral", "rel_abs"):
        eo_abs_log = np.log10(np.maximum(df_eo[ABSOLUTE_BAND_COLS].values, 1e-12))  # (64, 5)
        ec_abs_log = np.log10(np.maximum(df_ec[ABSOLUTE_BAND_COLS].values, 1e-12))  # (64, 5)
        X_mat = np.hstack([eo_rel, ec_rel, eo_abs_log, ec_abs_log])  # (64, 20)
    elif feat_type in ("full_spectral_3d", "spatial_spectral"):
        eo_abs_log = np.log10(np.maximum(df_eo[ABSOLUTE_BAND_COLS].values, 1e-12))  # (64, 5)
        ec_abs_log = np.log10(np.maximum(df_ec[ABSOLUTE_BAND_COLS].values, 1e-12))  # (64, 5)
        coords_3d = get_standard_3d_coordinates(ch_names)  # (64, 3)
        X_mat = np.hstack([eo_rel, ec_rel, eo_abs_log, ec_abs_log, coords_3d])  # (64, 23)
    elif feat_type == "single_eo":
        X_mat = eo_rel  # (64, 5)
    elif feat_type == "single_ec":
        X_mat = ec_rel  # (64, 5)
    else:
        raise ValueError(f"Unknown node_feature_type '{feat_type}'")

    # 3. Load Connectivity Adjacency (Stage 7)
    if isinstance(freq_band, (list, tuple)):
        mats = []
        for b in freq_band:
            conn_file = conn_root / f"stage7_{sub_lower}" / f"{subject_id}_R01_{b}_{conn_metric}.npy"
            if not conn_file.exists():
                raise FileNotFoundError(f"Connectivity matrix not found for {subject_id}: {conn_file}")
            mats.append(np.load(conn_file))
        adj_mat = np.mean(mats, axis=0)
    elif freq_band == "alpha_beta":
        alpha_file = conn_root / f"stage7_{sub_lower}" / f"{subject_id}_R01_alpha_{conn_metric}.npy"
        beta_file = conn_root / f"stage7_{sub_lower}" / f"{subject_id}_R01_beta_{conn_metric}.npy"
        if not alpha_file.exists() or not beta_file.exists():
            raise FileNotFoundError(f"Alpha/Beta connectivity files not found for {subject_id}")
        adj_mat = 0.5 * (np.load(alpha_file) + np.load(beta_file))
    else:
        conn_file = conn_root / f"stage7_{sub_lower}" / f"{subject_id}_R01_{freq_band}_{conn_metric}.npy"
        if not conn_file.exists():
            raise FileNotFoundError(f"Connectivity matrix not found for {subject_id}: {conn_file}")
        adj_mat = np.load(conn_file)  # (64, 64)
    n_nodes = adj_mat.shape[0]

    # 4. Sparsification & Edge Index Creation
    # Zero out diagonal
    np.fill_diagonal(adj_mat, 0.0)

    # Get upper triangle indices
    triu_i, triu_j = np.triu_indices(n_nodes, k=1)
    weights = adj_mat[triu_i, triu_j]

    if density < 1.0:
        total_possible = len(weights)
        k_keep = max(1, int(round(total_possible * density)))
        sorted_indices = np.argsort(weights)[::-1]
        top_k_indices = sorted_indices[:k_keep]

        selected_i = triu_i[top_k_indices]
        selected_j = triu_j[top_k_indices]

        # Check for isolated nodes
        if ensure_connected:
            degree_counts = np.zeros(n_nodes, dtype=int)
            for idx in top_k_indices:
                degree_counts[triu_i[idx]] += 1
                degree_counts[triu_j[idx]] += 1

            isolated = np.where(degree_counts == 0)[0]
            for iso_node in isolated:
                # Find strongest connection for iso_node
                row_weights = adj_mat[iso_node, :]
                best_neighbor = int(np.argmax(row_weights))
                if iso_node < best_neighbor:
                    selected_i = np.append(selected_i, iso_node)
                    selected_j = np.append(selected_j, best_neighbor)
                else:
                    selected_i = np.append(selected_i, best_neighbor)
                    selected_j = np.append(selected_j, iso_node)
    else:
        # Fully connected
        selected_i = triu_i
        selected_j = triu_j

    # Create symmetric undirected COO edge lists
    src = np.concatenate([selected_i, selected_j])
    dst = np.concatenate([selected_j, selected_i])
    edge_weights = adj_mat[src, dst]

    edge_index_tensor = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
    x_tensor = torch.tensor(X_mat, dtype=torch.float32)
    y_tensor = torch.tensor([y_val], dtype=torch.float32)

    data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_weight=edge_weight_tensor,
        y=y_tensor,
        num_nodes=n_nodes,
    )

    data.subject_id = subject_id
    data.metadata = {
        "subject_id": subject_id,
        "connectivity_metric": conn_metric,
        "frequency_band": freq_band,
        "sparsification_density": density,
        "node_feature_type": feat_type,
        "num_edges": int(edge_index_tensor.shape[1]),
    }

    return data
