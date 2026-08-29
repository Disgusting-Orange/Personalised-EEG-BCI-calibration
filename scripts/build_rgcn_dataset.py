"""Build Multi-Relational Graph Dataset for RGCN (5 Frequency Bands: Delta, Theta, Alpha, Beta, Gamma).

Saves graph objects cleanly to outputs/graph_dataset/rgcn_multi_band_top20
without modifying any existing dataset files.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_rgcn")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ABSOLUTE_BAND_COLS = ["delta_abs", "theta_abs", "alpha_abs", "beta_abs", "gamma_abs"]
RELATIVE_BAND_COLS = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"]

def sparsify_adj(adj: np.ndarray, density: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Sparsify dense symmetric adjacency matrix by retaining top-K percentile edges."""
    N = adj.shape[0]
    np.fill_diagonal(adj, 0.0)
    
    # Extract upper triangle entries
    triu_indices = np.triu_indices(N, k=1)
    weights = adj[triu_indices]
    
    num_edges_to_keep = max(1, int(round(len(weights) * density)))
    threshold = np.partition(weights, -num_edges_to_keep)[-num_edges_to_keep]
    
    adj_sparse = np.where(adj >= threshold, adj, 0.0)
    np.fill_diagonal(adj_sparse, 0.0)
    
    row, col = np.where(adj_sparse > 0)
    edge_weight = adj_sparse[row, col]
    edge_index = np.vstack([row, col])
    
    return edge_index, edge_weight

def build_single_rgcn_graph(subject_id: str, density: float = 0.20) -> Data:
    """Build multi-relational graph object for a single subject."""
    sub_lower = subject_id.lower()
    targets_root = REPOSITORY_ROOT / "outputs" / "targets"
    features_root = REPOSITORY_ROOT / "outputs" / "features"
    conn_root = REPOSITORY_ROOT / "outputs" / "connectivity"
    
    # 1. Target (Stage 3)
    target_file = targets_root / f"stage3_{sub_lower}" / "mi_targets.csv"
    df_target = pd.read_csv(target_file)
    y_val = float(df_target["balanced_accuracy"].iloc[0])
    
    # 2. Node Features (Stage 6) - 20 full spectral features
    eo_file = features_root / f"stage6_{sub_lower}" / f"{subject_id}_R01_band_powers.csv"
    ec_file = features_root / f"stage6_{sub_lower}" / f"{subject_id}_R02_band_powers.csv"
    
    df_eo = pd.read_csv(eo_file, index_col=0)
    df_ec = pd.read_csv(ec_file, index_col=0)
    
    eo_rel = df_eo[RELATIVE_BAND_COLS].values
    ec_rel = df_ec[RELATIVE_BAND_COLS].values
    eo_abs_log = np.log10(np.maximum(df_eo[ABSOLUTE_BAND_COLS].values, 1e-12))
    ec_abs_log = np.log10(np.maximum(df_ec[ABSOLUTE_BAND_COLS].values, 1e-12))
    
    X_mat = np.hstack([eo_rel, ec_rel, eo_abs_log, ec_abs_log]) # (64, 20)
    
    # 3. Multi-Relational Edges (Stage 7 - 5 Frequency Bands)
    all_edge_indices = []
    all_edge_weights = []
    all_edge_types = []
    
    for r_idx, band in enumerate(BANDS):
        conn_file = conn_root / f"stage7_{sub_lower}" / f"{subject_id}_R01_{band}_wpli.npy"
        adj = np.load(conn_file)
        
        edge_index, edge_weight = sparsify_adj(adj, density=density)
        edge_type = np.full(edge_index.shape[1], r_idx, dtype=np.int64)
        
        all_edge_indices.append(edge_index)
        all_edge_weights.append(edge_weight)
        all_edge_types.append(edge_type)
        
    edge_index_cat = np.hstack(all_edge_indices)
    edge_weight_cat = np.concatenate(all_edge_weights)
    edge_type_cat = np.concatenate(all_edge_types)
    
    data = Data(
        x=torch.tensor(X_mat, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_cat, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight_cat, dtype=torch.float32),
        edge_type=torch.tensor(edge_type_cat, dtype=torch.long),
        y=torch.tensor([y_val], dtype=torch.float32),
        subject_id=subject_id
    )
    return data

def build_full_rgcn_dataset(out_dir: str = "outputs/graph_dataset/rgcn_multi_band_top20"):
    """Construct and collate RGCN dataset for all 109 subjects."""
    out_path = REPOSITORY_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    
    data_list = []
    logger.info("Building multi-relational RGCN graphs for 109 subjects...")
    
    for sub_idx in range(1, 110):
        sub_id = f"S{sub_idx:03d}"
        data = build_single_rgcn_graph(sub_id, density=0.20)
        data_list.append(data)
        
    torch.save(data_list, out_path / "data_list.pt")
    logger.info(f"Successfully saved {len(data_list)} RGCN graph objects to {out_path / 'data_list.pt'}")

if __name__ == "__main__":
    build_full_rgcn_dataset()
