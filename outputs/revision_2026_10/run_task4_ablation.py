"""
Task 4 — Graph Ablation Suite under 109-subject LOSO CV.

Evaluates:
(i)   Random Forest and Ridge on vectorized upper-triangle alpha wPLI (2,016-D)
(ii)  Primary GCN with identity adjacency (self-loops only, no edges)
(iii) Primary GCN with node-permuted / shuffled adjacency
(iv)  Node-feature MLP on flattened 64x10 node features (standardized, fair capacity)
(v)   Primary GCN over N random initialization seeds (N = 5: seeds 42, 100, 2024, 777, 999)

Computes point estimates, bootstrap CIs, paired differences vs Primary GCN,
and writes Table tab:ablation rows and CSVs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("task4_ablation")

REPO_ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = REPO_ROOT / "outputs" / "features"
CONN_DIR = REPO_ROOT / "outputs" / "connectivity"
OUT_DIR = REPO_ROOT / "outputs" / "revision_2026_10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metrics & Bootstrap Helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-9 else 0.0
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    mae = float(100.0 * np.mean(np.abs(err)))
    rmse = float(100.0 * np.sqrt(np.mean(err ** 2)))
    return {"r": r, "R2": r2, "MAE": mae, "RMSE": rmse}

def bootstrap_ci_and_diff(y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray], ref_key: str = "primary_gcn", n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    
    boot_metrics = {k: [compute_metrics(y_true[b], y_pred_dict[k][b]) for b in idx] for k in y_pred_dict}
    
    ci_table = {}
    for k in y_pred_dict:
        pt = compute_metrics(y_true, y_pred_dict[k])
        ci_table[k] = {"point": pt}
        for m in ["r", "R2", "MAE", "RMSE"]:
            vals = [b[m] for b in boot_metrics[k]]
            ci_table[k][f"{m}_ci"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
            
    paired_table = {}
    ref_boot = boot_metrics[ref_key]
    ref_pt = compute_metrics(y_true, y_pred_dict[ref_key])
    
    for k in y_pred_dict:
        if k == ref_key:
            continue
        pt_diff = {m: compute_metrics(y_true, y_pred_dict[k])[m] - ref_pt[m] for m in ["r", "R2", "MAE", "RMSE"]}
        paired_table[k] = {"point_diff": pt_diff}
        for m in ["R2", "MAE", "RMSE"]:
            diffs = [b[m] - ref_boot[i][m] for i, b in enumerate(boot_metrics[k])]
            paired_table[k][f"{m}_diff_ci"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
            
    return ci_table, paired_table

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCNRegressor(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)

class MLPRegressor(nn.Module):
    def __init__(self, in_features: int = 640, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc3(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data():
    df_gt = pd.read_csv(REPO_ROOT / "LOSO_OOF_predictions.csv")
    y_actual = df_gt["actual_mi_accuracy"].values
    sub_ids = df_gt["subject_id"].values
    N = len(sub_ids)
    
    relative_cols = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"]
    triu_i, triu_j = np.triu_indices(64, k=1)
    
    X_node_list = []
    X_flat_list = []
    W_upper_list = []
    adj_mats = []
    
    for i in range(1, 110):
        sub = f"S{i:03d}"
        sub_l = sub.lower()
        f_eo = FEAT_DIR / f"stage6_{sub_l}" / f"{sub}_R01_band_powers.csv"
        f_ec = FEAT_DIR / f"stage6_{sub_l}" / f"{sub}_R02_band_powers.csv"
        df_eo = pd.read_csv(f_eo, index_col=0)
        df_ec = pd.read_csv(f_ec, index_col=0)
        eo_rel = df_eo[relative_cols].values # (64, 5)
        ec_rel = df_ec[relative_cols].values # (64, 5)
        X_mat = np.hstack([eo_rel, ec_rel]) # (64, 10)
        X_node_list.append(X_mat)
        X_flat_list.append(X_mat.flatten()) # (640,)
        
        w1 = np.load(CONN_DIR / f"stage7_{sub_l}" / f"{sub}_R01_alpha_wpli.npy")
        w2 = np.load(CONN_DIR / f"stage7_{sub_l}" / f"{sub}_R02_alpha_wpli.npy")
        w_joint = 0.5 * (w1 + w2)
        adj_mats.append(w_joint)
        W_upper_list.append(w_joint[triu_i, triu_j]) # (2016,)
        
    X_node = np.array(X_node_list, dtype=np.float32) # (109, 64, 10)
    X_flat = np.array(X_flat_list, dtype=np.float32) # (109, 640)
    X_wpli_vec = np.array(W_upper_list, dtype=np.float32) # (109, 2016)
    
    return sub_ids, y_actual, X_node, X_flat, X_wpli_vec, adj_mats

# ---------------------------------------------------------------------------
# Graph Builders
# ---------------------------------------------------------------------------

def build_sparse_top15_graph(X_mat, adj_mat, y_val):
    n_nodes = 64
    adj = adj_mat.copy()
    np.fill_diagonal(adj, 0.0)
    triu_i, triu_j = np.triu_indices(n_nodes, k=1)
    weights = adj[triu_i, triu_j]
    k_keep = max(1, int(round(len(weights) * 0.15)))
    sorted_idx = np.argsort(weights)[::-1]
    top_k = sorted_idx[:k_keep]
    selected_i = triu_i[top_k]
    selected_j = triu_j[top_k]
    
    degree_counts = np.zeros(n_nodes, dtype=int)
    for idx in top_k:
        degree_counts[triu_i[idx]] += 1
        degree_counts[triu_j[idx]] += 1
    isolated = np.where(degree_counts == 0)[0]
    for iso_node in isolated:
        row_w = adj[iso_node, :]
        best_neighbor = int(np.argmax(row_w))
        if iso_node < best_neighbor:
            selected_i = np.append(selected_i, iso_node)
            selected_j = np.append(selected_j, best_neighbor)
        else:
            selected_i = np.append(selected_i, best_neighbor)
            selected_j = np.append(selected_j, iso_node)
            
    src = np.concatenate([selected_i, selected_j])
    dst = np.concatenate([selected_j, selected_i])
    edge_w = adj[src, dst]
    return Data(
        x=torch.tensor(X_mat, dtype=torch.float32),
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_weight=torch.tensor(edge_w, dtype=torch.float32),
        y=torch.tensor([y_val], dtype=torch.float32),
        num_nodes=n_nodes
    )

def build_identity_graph(X_mat, y_val):
    n_nodes = 64
    src = np.arange(n_nodes)
    dst = np.arange(n_nodes)
    edge_w = np.ones(n_nodes, dtype=np.float32)
    return Data(
        x=torch.tensor(X_mat, dtype=torch.float32),
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_weight=torch.tensor(edge_w, dtype=torch.float32),
        y=torch.tensor([y_val], dtype=torch.float32),
        num_nodes=n_nodes
    )

def build_shuffled_graph(X_mat, adj_mat, y_val, seed=42):
    n_nodes = 64
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_nodes)
    shuffled_adj = adj_mat[perm, :][:, perm]
    return build_sparse_top15_graph(X_mat, shuffled_adj, y_val)

# ---------------------------------------------------------------------------
# GCN Training Functions (Parallelized across LOSO folds)
# ---------------------------------------------------------------------------

def train_single_gcn_fold(i: int, graph_list: List[Data], base_seed: int = 42) -> float:
    device = torch.device("cpu")
    test_sub = graph_list[i]
    train_pool = [graph_list[j] for j in range(len(graph_list)) if j != i]
    
    rng = np.random.RandomState(base_seed + i)
    perm = rng.permutation(len(train_pool))
    val_size = 11
    val_indices = perm[:val_size]
    tr_indices = perm[val_size:]
    
    train_set = [train_pool[idx] for idx in tr_indices]
    val_set = [train_pool[idx] for idx in val_indices]
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader([test_sub], batch_size=1, shuffle=False)
    
    torch.manual_seed(base_seed + i)
    model = GCNRegressor(in_channels=10, hidden_channels=64, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    
    for epoch in range(150):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).view(-1)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).view(-1)
                val_loss += F.mse_loss(pred, batch.y.view(-1)).item() * batch.num_graphs
        val_loss /= len(val_set)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break
                
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            p_val = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            raw_val = float(p_val.item())
    return raw_val

def train_gcn_loso(graph_list: List[Data], base_seed: int = 42) -> np.ndarray:
    N = len(graph_list)
    preds = np.zeros(N)
    for i in range(N):
        preds[i] = train_single_gcn_fold(i, graph_list, base_seed)
        if (i + 1) % 25 == 0 or (i + 1) == N:
            logger.info("  GCN (seed %d) fold %d/%d completed.", base_seed, i + 1, N)
    return preds

# ---------------------------------------------------------------------------
# MLP Training Functions (Parallelized across LOSO folds)
# ---------------------------------------------------------------------------

def train_single_mlp_fold(i: int, X_flat: np.ndarray, y_actual: np.ndarray, base_seed: int = 42) -> float:
    device = torch.device("cpu")
    N = len(y_actual)
    train_idx = [j for j in range(N) if j != i]
    X_train_val = X_flat[train_idx]
    y_train_val = y_actual[train_idx]
    X_test = X_flat[[i]]
    
    rng = np.random.RandomState(base_seed + i)
    perm = rng.permutation(len(train_idx))
    val_size = 11
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_val[tr_idx])
    X_va = scaler.transform(X_train_val[val_idx])
    X_te = scaler.transform(X_test)
    
    y_tr = y_train_val[tr_idx]
    y_va = y_train_val[val_idx]
    
    tr_dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    va_dataset = torch.utils.data.TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(va_dataset, batch_size=16, shuffle=False)
    
    torch.manual_seed(base_seed + i)
    model = MLPRegressor(in_features=640, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    
    for epoch in range(150):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                val_loss += F.mse_loss(pred, by).item() * len(by)
        val_loss /= len(va_dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break
                
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    model.eval()
    with torch.no_grad():
        raw_val = float(model(torch.tensor(X_te, dtype=torch.float32)).item())
    return raw_val

def train_mlp_loso(X_flat: np.ndarray, y_actual: np.ndarray, base_seed: int = 42) -> np.ndarray:
    N = len(y_actual)
    preds = np.zeros(N)
    for i in range(N):
        preds[i] = train_single_mlp_fold(i, X_flat, y_actual, base_seed)
        if (i + 1) % 25 == 0 or (i + 1) == N:
            logger.info("  MLP fold %d/%d completed.", i + 1, N)
    return preds

# ---------------------------------------------------------------------------
# Tabular Baseline Evaluators on 2016-D wPLI
# ---------------------------------------------------------------------------

def _fit_single_rf_fold(i: int, X_wpli: np.ndarray, y_actual: np.ndarray) -> float:
    N = len(y_actual)
    train_idx = [j for j in range(N) if j != i]
    rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42 + i, n_jobs=1)
    rf.fit(X_wpli[train_idx], y_actual[train_idx])
    return float(rf.predict(X_wpli[[i]])[0])

def run_rf_wpli(X_wpli: np.ndarray, y_actual: np.ndarray) -> np.ndarray:
    N = len(y_actual)
    preds = np.zeros(N)
    for i in range(N):
        preds[i] = _fit_single_rf_fold(i, X_wpli, y_actual)
    return preds

def _fit_single_ridge_fold(i: int, split: Dict[str, Any], X_wpli: np.ndarray, y_actual: np.ndarray) -> float:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    X_tr, y_tr = X_wpli[train_idx], y_actual[train_idx]
    pipe = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(random_state=42 + i))])
    cv = KFold(n_splits=5, shuffle=True, random_state=42 + i)
    grid = {"regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(X_tr, y_tr)
    return float(search.predict(X_wpli[test_idx])[0])

def run_ridge_wpli(X_wpli: np.ndarray, y_actual: np.ndarray, splits: List[Dict[str, Any]]) -> np.ndarray:
    N = len(y_actual)
    preds = np.zeros(N)
    for i in range(N):
        preds[i] = _fit_single_ridge_fold(i, splits[i], X_wpli, y_actual)
    return preds

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    logger.info("Starting Task 4 — Graph Ablation Suite...")
    
    sub_ids, y_actual, X_node, X_flat, X_wpli_vec, adj_mats = load_data()
    N = len(y_actual)
    
    with open(REPO_ROOT / "outputs" / "benchmark" / "loso_splits.json") as fp:
        splits = json.load(fp)
        
    predictions = {}
    
    # 0. Primary GCN reference (from frozen ledger)
    df_ledger = pd.read_csv(REPO_ROOT / "LOSO_OOF_predictions.csv")
    predictions["primary_gcn"] = df_ledger["pred_base_gcn_alpha"].values
    
    # Check if primary GCN matches run_corrected_r04_r14_pipeline
    m_ref = compute_metrics(y_actual, predictions["primary_gcn"])
    logger.info("Primary GCN Reference: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_ref["r"], m_ref["R2"], m_ref["MAE"], m_ref["RMSE"])
    
    # (i) Random Forest & Ridge on vectorized 2016-D wPLI
    logger.info("[1/5] Evaluating Random Forest on vectorized 2016-D wPLI...")
    predictions["rf_wpli"] = run_rf_wpli(X_wpli_vec, y_actual)
    m_rf = compute_metrics(y_actual, predictions["rf_wpli"])
    logger.info("  RF on 2016-D wPLI: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_rf["r"], m_rf["R2"], m_rf["MAE"], m_rf["RMSE"])
    
    logger.info("[2/5] Evaluating Ridge on vectorized 2016-D wPLI...")
    predictions["ridge_wpli"] = run_ridge_wpli(X_wpli_vec, y_actual, splits)
    m_ridge = compute_metrics(y_actual, predictions["ridge_wpli"])
    logger.info("  Ridge on 2016-D wPLI: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_ridge["r"], m_ridge["R2"], m_ridge["MAE"], m_ridge["RMSE"])
    
    # (iv) MLP on flattened 64x10 node features (640-D)
    logger.info("[3/5] Evaluating Node-Feature MLP (no graph, 640-D)...")
    predictions["mlp_no_graph"] = train_mlp_loso(X_flat, y_actual, base_seed=42)
    m_mlp = compute_metrics(y_actual, predictions["mlp_no_graph"])
    logger.info("  Node-Feature MLP: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_mlp["r"], m_mlp["R2"], m_mlp["MAE"], m_mlp["RMSE"])
    
    # (ii) GCN with identity adjacency
    logger.info("[4/5] Evaluating Primary GCN with Identity Adjacency (no edges)...")
    # Check if we use exact results from negative_control_results.mat / topology_ablation.csv or rerun
    # Let's run identity graph
    graphs_identity = [build_identity_graph(X_node[i], y_actual[i]) for i in range(N)]
    predictions["gcn_identity"] = train_gcn_loso(graphs_identity, base_seed=42)
    m_id = compute_metrics(y_actual, predictions["gcn_identity"])
    logger.info("  GCN Identity Adjacency: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_id["r"], m_id["R2"], m_id["MAE"], m_id["RMSE"])
    
    # (iii) GCN with shuffled adjacency
    logger.info("[5/5] Evaluating Primary GCN with Shuffled Adjacency...")
    graphs_shuffled = [build_shuffled_graph(X_node[i], adj_mats[i], y_actual[i], seed=42 + i) for i in range(N)]
    predictions["gcn_shuffled"] = train_gcn_loso(graphs_shuffled, base_seed=42)
    m_shuf = compute_metrics(y_actual, predictions["gcn_shuffled"])
    logger.info("  GCN Shuffled Adjacency: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", m_shuf["r"], m_shuf["R2"], m_shuf["MAE"], m_shuf["RMSE"])
    
    # (v) Primary GCN over N seeds (N = 5)
    seeds = [42, 100, 2024, 777, 999]
    seed_metrics = []
    logger.info("[Seed Stability] Evaluating Primary GCN across %d random seeds: %s...", len(seeds), seeds)
    graphs_primary = [build_sparse_top15_graph(X_node[i], adj_mats[i], y_actual[i]) for i in range(N)]
    
    for s_idx, s in enumerate(seeds):
        if s == 42:
            preds_s = predictions["primary_gcn"]
        else:
            preds_s = train_gcn_loso(graphs_primary, base_seed=s)
        predictions[f"primary_gcn_seed_{s}"] = preds_s
        m_s = compute_metrics(y_actual, preds_s)
        seed_metrics.append(m_s)
        logger.info("  Seed %d: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", s, m_s["r"], m_s["R2"], m_s["MAE"], m_s["RMSE"])
        
    df_seed_metrics = pd.DataFrame(seed_metrics)
    seed_summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "r_mean": float(df_seed_metrics["r"].mean()),
        "r_sd": float(df_seed_metrics["r"].std()),
        "R2_mean": float(df_seed_metrics["R2"].mean()),
        "R2_sd": float(df_seed_metrics["R2"].std()),
        "MAE_mean": float(df_seed_metrics["MAE"].mean()),
        "MAE_sd": float(df_seed_metrics["MAE"].std()),
        "RMSE_mean": float(df_seed_metrics["RMSE"].mean()),
        "RMSE_sd": float(df_seed_metrics["RMSE"].std()),
    }
    logger.info("Seed Summary (N=%d): r=%.4f +/- %.4f, R2=%+.4f +/- %.4f, MAE=%.2f +/- %.2f, RMSE=%.2f +/- %.2f",
                len(seeds), seed_summary["r_mean"], seed_summary["r_sd"], seed_summary["R2_mean"], seed_summary["R2_sd"],
                seed_summary["MAE_mean"], seed_summary["MAE_sd"], seed_summary["RMSE_mean"], seed_summary["RMSE_sd"])
                
    # Save OOF predictions
    df_preds_out = pd.DataFrame({"subject_id": sub_ids, "actual_mi_accuracy": y_actual})
    for k in predictions:
        df_preds_out[k] = predictions[k]
    df_preds_out.to_csv(OUT_DIR / "ablation_oof_predictions.csv", index=False)
    
    # Compute Bootstrap CIs and Paired Differences
    ci_table, paired_table = bootstrap_ci_and_diff(y_actual, predictions, ref_key="primary_gcn", n_boot=1000, seed=0)
    
    # Build Table tab:ablation rows
    ablation_rows = [
        {
            "Model": "Primary GCN (reference)",
            "r": f"{m_ref['r']:.4f}",
            "R2": f"{m_ref['R2']:+.4f}",
            "MAE": f"{m_ref['MAE']:.2f}",
            "RMSE": f"{m_ref['RMSE']:.2f}",
            "r_ci": f"[{ci_table['primary_gcn']['r_ci'][0]:.4f}, {ci_table['primary_gcn']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['primary_gcn']['R2_ci'][0]:+.4f}, {ci_table['primary_gcn']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['primary_gcn']['MAE_ci'][0]:.2f}, {ci_table['primary_gcn']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['primary_gcn']['RMSE_ci'][0]:.2f}, {ci_table['primary_gcn']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": "--",
            "Delta_MAE": "--",
            "Delta_RMSE": "--"
        },
        {
            "Model": f"Primary GCN, {len(seeds)} seeds",
            "r": f"{seed_summary['r_mean']:.4f} $\\pm$ {seed_summary['r_sd']:.4f}",
            "R2": f"{seed_summary['R2_mean']:+.4f} $\\pm$ {seed_summary['R2_sd']:.4f}",
            "MAE": f"{seed_summary['MAE_mean']:.2f} $\\pm$ {seed_summary['MAE_sd']:.2f}",
            "RMSE": f"{seed_summary['RMSE_mean']:.2f} $\\pm$ {seed_summary['RMSE_sd']:.2f}",
            "r_ci": "--",
            "R2_ci": "--",
            "MAE_ci": "--",
            "RMSE_ci": "--",
            "Delta_R2": "--",
            "Delta_MAE": "--",
            "Delta_RMSE": "--"
        },
        {
            "Model": "GCN, identity adjacency",
            "r": f"{m_id['r']:.4f}",
            "R2": f"{m_id['R2']:+.4f}",
            "MAE": f"{m_id['MAE']:.2f}",
            "RMSE": f"{m_id['RMSE']:.2f}",
            "r_ci": f"[{ci_table['gcn_identity']['r_ci'][0]:.4f}, {ci_table['gcn_identity']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['gcn_identity']['R2_ci'][0]:+.4f}, {ci_table['gcn_identity']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['gcn_identity']['MAE_ci'][0]:.2f}, {ci_table['gcn_identity']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['gcn_identity']['RMSE_ci'][0]:.2f}, {ci_table['gcn_identity']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": f"{paired_table['gcn_identity']['point_diff']['R2']:+.4f} [{paired_table['gcn_identity']['R2_diff_ci'][0]:+.4f}, {paired_table['gcn_identity']['R2_diff_ci'][1]:+.4f}]",
            "Delta_MAE": f"{paired_table['gcn_identity']['point_diff']['MAE']:+.2f} [{paired_table['gcn_identity']['MAE_diff_ci'][0]:+.2f}, {paired_table['gcn_identity']['MAE_diff_ci'][1]:+.2f}]",
            "Delta_RMSE": f"{paired_table['gcn_identity']['point_diff']['RMSE']:+.2f} [{paired_table['gcn_identity']['RMSE_diff_ci'][0]:+.2f}, {paired_table['gcn_identity']['RMSE_diff_ci'][1]:+.2f}]"
        },
        {
            "Model": "GCN, shuffled adjacency",
            "r": f"{m_shuf['r']:.4f}",
            "R2": f"{m_shuf['R2']:+.4f}",
            "MAE": f"{m_shuf['MAE']:.2f}",
            "RMSE": f"{m_shuf['RMSE']:.2f}",
            "r_ci": f"[{ci_table['gcn_shuffled']['r_ci'][0]:.4f}, {ci_table['gcn_shuffled']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['gcn_shuffled']['R2_ci'][0]:+.4f}, {ci_table['gcn_shuffled']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['gcn_shuffled']['MAE_ci'][0]:.2f}, {ci_table['gcn_shuffled']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['gcn_shuffled']['RMSE_ci'][0]:.2f}, {ci_table['gcn_shuffled']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": f"{paired_table['gcn_shuffled']['point_diff']['R2']:+.4f} [{paired_table['gcn_shuffled']['R2_diff_ci'][0]:+.4f}, {paired_table['gcn_shuffled']['R2_diff_ci'][1]:+.4f}]",
            "Delta_MAE": f"{paired_table['gcn_shuffled']['point_diff']['MAE']:+.2f} [{paired_table['gcn_shuffled']['MAE_diff_ci'][0]:+.2f}, {paired_table['gcn_shuffled']['MAE_diff_ci'][1]:+.2f}]",
            "Delta_RMSE": f"{paired_table['gcn_shuffled']['point_diff']['RMSE']:+.2f} [{paired_table['gcn_shuffled']['RMSE_diff_ci'][0]:+.2f}, {paired_table['gcn_shuffled']['RMSE_diff_ci'][1]:+.2f}]"
        },
        {
            "Model": "Node-feature MLP (no graph)",
            "r": f"{m_mlp['r']:.4f}",
            "R2": f"{m_mlp['R2']:+.4f}",
            "MAE": f"{m_mlp['MAE']:.2f}",
            "RMSE": f"{m_mlp['RMSE']:.2f}",
            "r_ci": f"[{ci_table['mlp_no_graph']['r_ci'][0]:.4f}, {ci_table['mlp_no_graph']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['mlp_no_graph']['R2_ci'][0]:+.4f}, {ci_table['mlp_no_graph']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['mlp_no_graph']['MAE_ci'][0]:.2f}, {ci_table['mlp_no_graph']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['mlp_no_graph']['RMSE_ci'][0]:.2f}, {ci_table['mlp_no_graph']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": f"{paired_table['mlp_no_graph']['point_diff']['R2']:+.4f} [{paired_table['mlp_no_graph']['R2_diff_ci'][0]:+.4f}, {paired_table['mlp_no_graph']['R2_diff_ci'][1]:+.4f}]",
            "Delta_MAE": f"{paired_table['mlp_no_graph']['point_diff']['MAE']:+.2f} [{paired_table['mlp_no_graph']['MAE_diff_ci'][0]:+.2f}, {paired_table['mlp_no_graph']['MAE_diff_ci'][1]:+.2f}]",
            "Delta_RMSE": f"{paired_table['mlp_no_graph']['point_diff']['RMSE']:+.2f} [{paired_table['mlp_no_graph']['RMSE_diff_ci'][0]:+.2f}, {paired_table['mlp_no_graph']['RMSE_diff_ci'][1]:+.2f}]"
        },
        {
            "Model": "Random forest, vectorized wPLI",
            "r": f"{m_rf['r']:.4f}",
            "R2": f"{m_rf['R2']:+.4f}",
            "MAE": f"{m_rf['MAE']:.2f}",
            "RMSE": f"{m_rf['RMSE']:.2f}",
            "r_ci": f"[{ci_table['rf_wpli']['r_ci'][0]:.4f}, {ci_table['rf_wpli']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['rf_wpli']['R2_ci'][0]:+.4f}, {ci_table['rf_wpli']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['rf_wpli']['MAE_ci'][0]:.2f}, {ci_table['rf_wpli']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['rf_wpli']['RMSE_ci'][0]:.2f}, {ci_table['rf_wpli']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": f"{paired_table['rf_wpli']['point_diff']['R2']:+.4f} [{paired_table['rf_wpli']['R2_diff_ci'][0]:+.4f}, {paired_table['rf_wpli']['R2_diff_ci'][1]:+.4f}]",
            "Delta_MAE": f"{paired_table['rf_wpli']['point_diff']['MAE']:+.2f} [{paired_table['rf_wpli']['MAE_diff_ci'][0]:+.2f}, {paired_table['rf_wpli']['MAE_diff_ci'][1]:+.2f}]",
            "Delta_RMSE": f"{paired_table['rf_wpli']['point_diff']['RMSE']:+.2f} [{paired_table['rf_wpli']['RMSE_diff_ci'][0]:+.2f}, {paired_table['rf_wpli']['RMSE_diff_ci'][1]:+.2f}]"
        },
        {
            "Model": "Ridge, vectorized wPLI",
            "r": f"{m_ridge['r']:.4f}",
            "R2": f"{m_ridge['R2']:+.4f}",
            "MAE": f"{m_ridge['MAE']:.2f}",
            "RMSE": f"{m_ridge['RMSE']:.2f}",
            "r_ci": f"[{ci_table['ridge_wpli']['r_ci'][0]:.4f}, {ci_table['ridge_wpli']['r_ci'][1]:.4f}]",
            "R2_ci": f"[{ci_table['ridge_wpli']['R2_ci'][0]:+.4f}, {ci_table['ridge_wpli']['R2_ci'][1]:+.4f}]",
            "MAE_ci": f"[{ci_table['ridge_wpli']['MAE_ci'][0]:.2f}, {ci_table['ridge_wpli']['MAE_ci'][1]:.2f}]",
            "RMSE_ci": f"[{ci_table['ridge_wpli']['RMSE_ci'][0]:.2f}, {ci_table['ridge_wpli']['RMSE_ci'][1]:.2f}]",
            "Delta_R2": f"{paired_table['ridge_wpli']['point_diff']['R2']:+.4f} [{paired_table['ridge_wpli']['R2_diff_ci'][0]:+.4f}, {paired_table['ridge_wpli']['R2_diff_ci'][1]:+.4f}]",
            "Delta_MAE": f"{paired_table['ridge_wpli']['point_diff']['MAE']:+.2f} [{paired_table['ridge_wpli']['MAE_diff_ci'][0]:+.2f}, {paired_table['ridge_wpli']['MAE_diff_ci'][1]:+.2f}]",
            "Delta_RMSE": f"{paired_table['ridge_wpli']['point_diff']['RMSE']:+.2f} [{paired_table['ridge_wpli']['RMSE_diff_ci'][0]:+.2f}, {paired_table['ridge_wpli']['RMSE_diff_ci'][1]:+.2f}]"
        }
    ]
    
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(OUT_DIR / "table_ablation_results.csv", index=False)
    
    latex_lines = []
    for r in ablation_rows:
        latex_lines.append(f"{r['Model']} & {r['r']} & {r['R2']} & {r['MAE']} & {r['RMSE']} \\\\")
        
    with open(OUT_DIR / "table_ablation_rows.tex", "w") as fp:
        fp.write("\n".join(latex_lines) + "\n")
        
    with open(OUT_DIR / "seed_stability_summary.json", "w") as fp:
        json.dump(seed_summary, fp, indent=2)
        
    print("\n" + "=" * 80)
    print("TABLE ABLATION RESULTS:")
    print("=" * 80)
    for line in latex_lines:
        print(line)
        
    elapsed = time.time() - t_start
    logger.info("Task 4 completed in %.2f seconds.", elapsed)

if __name__ == "__main__":
    main()
