"""Systematic Audit and Architecture/Loss Tuning for GCN under strict LOSO.

Evaluates:
- Node feature normalization using training subjects only
- Architectures: hidden=32, 64; layers=1, 2, 3
- Loss functions: mse, huber, variance_mse
- Evaluation metrics: MAE, RMSE, Pearson r, Spearman rho, R^2 (both raw and fold-calibrated)
"""

import os
import sys
from pathlib import Path

# Add repository root to sys.path so 'src' is resolvable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.graph.dataset import EEGGraphDataset
from src.graph.gcn_model import GCNRegressor

def normalize_graphs(train_data, val_data, test_data):
    """Normalize node features using training subjects only (zero leakage)."""
    # Concatenate all node features from training subjects
    all_train_x = torch.cat([d.x for d in train_data], dim=0) # (N_train * 64, in_dim)
    mu = all_train_x.mean(dim=0, keepdim=True)
    std = all_train_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    def apply_norm(g_list):
        out = []
        for g in g_list:
            g_c = g.clone()
            g_c.x = (g.x - mu) / std
            out.append(g_c)
        return out

    return apply_norm(train_data), apply_norm(val_data), apply_norm(test_data), mu, std

def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, loss_type="variance_mse", epochs=80, patience=15, device="cpu"):
    best_loss = float("inf")
    best_state = None
    patience_cnt = 0
    model.to(device)

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).view(-1)
            by = batch.y.view(-1)

            if loss_type == "variance_mse" and pred.size(0) > 1:
                mse = F.mse_loss(pred, by)
                var_p = torch.var(pred, unbiased=False)
                var_t = torch.var(by, unbiased=False)
                loss = mse + 0.5 * torch.abs(var_p - var_t)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(pred, by, beta=0.05)
            else:
                loss = F.mse_loss(pred, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).view(-1)
                by = batch.y.view(-1)
                if loss_type == "huber":
                    l = F.smooth_l1_loss(pred, by, beta=0.05)
                else:
                    l = F.mse_loss(pred, by)
                val_loss += l.item() * batch.num_graphs
                n_val += batch.num_graphs

        val_loss /= max(1, n_val)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_config(ds, hidden_dim, num_layers, loss_type, lr=0.005, epochs=80, patience=15, seed=42):
    n_sub = len(ds)
    y_actual = np.array([float(ds[i].y.item()) for i in range(n_sub)])
    y_pred_raw = np.zeros(n_sub)
    y_pred_fold_cal = np.zeros(n_sub)

    t0 = time.time()
    for i in range(n_sub):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        test_data = [ds[i]]
        train_val_data = [ds[j] for j in range(n_sub) if j != i]

        # 90-10 split
        n_tv = len(train_val_data)
        n_val = max(1, int(n_tv * 0.1))
        val_raw = train_val_data[:n_val]
        train_raw = train_val_data[n_val:]

        # Normalize features strictly from training subjects
        train_norm, val_norm, test_norm, _, _ = normalize_graphs(train_raw, val_raw, test_data)

        train_loader = DataLoader(train_norm, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_norm, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_norm, batch_size=1, shuffle=False)

        in_dim = int(ds[0].x.shape[1])
        model = GCNRegressor(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layers, dropout=0.2, pooling="mean")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)

        model = train_one_fold(model, train_loader, val_loader, optimizer, scheduler, loss_type=loss_type, epochs=epochs, patience=patience)

        # Predict held-out test subject
        model.eval()
        with torch.no_grad():
            for b in test_loader:
                raw_val = float(model(b.x, b.edge_index, b.edge_weight, b.batch).item())
                y_pred_raw[i] = raw_val

        # Also get predictions on training set to fit fold-nested calibration
        train_preds = []
        train_targs = []
        with torch.no_grad():
            for g in train_norm:
                p = float(model(g.x, g.edge_index, g.edge_weight, g.batch).item())
                train_preds.append(p)
                train_targs.append(float(g.y.item()))
        train_preds = np.array(train_preds)
        train_targs = np.array(train_targs)

        slope_i, int_i, _, _, _ = stats.linregress(train_preds, train_targs)
        if np.isnan(slope_i) or slope_i <= 0:
            slope_i, int_i = 1.0, 0.0
        y_pred_fold_cal[i] = slope_i * raw_val + int_i

    runtime = time.time() - t0

    def calc_metrics(yp):
        mae = float(np.mean(np.abs(y_actual - yp)))
        rmse = float(np.sqrt(np.mean((y_actual - yp)**2)))
        ss_tot = float(np.sum((y_actual - np.mean(y_actual))**2))
        ss_res = float(np.sum((y_actual - yp)**2))
        r2 = float(1.0 - ss_res / ss_tot)
        r, p = stats.pearsonr(y_actual, yp)
        rho, rho_p = stats.spearmanr(y_actual, yp)
        return mae, rmse, r2, r, p, rho, rho_p

    m_raw = calc_metrics(y_pred_raw)
    m_cal = calc_metrics(y_pred_fold_cal)

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "loss_type": loss_type,
        "runtime_sec": round(runtime, 1),
        # Raw Metrics
        "raw_mae": round(m_raw[0], 4),
        "raw_rmse": round(m_raw[1], 4),
        "raw_r2": round(m_raw[2], 4),
        "raw_pearson_r": round(m_raw[3], 4),
        "raw_pearson_p": m_raw[4],
        "raw_spearman_rho": round(m_raw[5], 4),
        # Leakage-Free Fold-Calibrated Metrics
        "cal_mae": round(m_cal[0], 4),
        "cal_rmse": round(m_cal[1], 4),
        "cal_r2": round(m_cal[2], 4),
        "cal_pearson_r": round(m_cal[3], 4),
        "cal_pearson_p": m_cal[4],
        "cal_spearman_rho": round(m_cal[5], 4),
    }

def main():
    print("Loading graph dataset wpli_alpha_full_spectral_top20...")
    ds = EEGGraphDataset("outputs/graph_dataset/wpli_alpha_full_spectral_top20")
    print(f"Loaded {len(ds)} graphs with {ds[0].x.shape[1]} features.")

    configs = [
        # (hidden_dim, num_layers, loss_type)
        (32, 1, "variance_mse"),
        (32, 2, "variance_mse"),
        (64, 2, "variance_mse"),
        (64, 3, "variance_mse"),
        (64, 2, "mse"),
        (64, 2, "huber"),
    ]

    results = []
    for h, l, loss in configs:
        print(f"\n--- Testing GCN: hidden={h}, layers={l}, loss={loss} ---")
        res = evaluate_config(ds, hidden_dim=h, num_layers=l, loss_type=loss)
        print(f"Results: Raw r={res['raw_pearson_r']} (p={res['raw_pearson_p']:.4e}), Cal R2={res['cal_r2']}, Cal MAE={res['cal_mae']} (Runtime: {res['runtime_sec']}s)")
        results.append(res)

    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/gcn_architecture_loss_ablation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved systematic ablation report to {out_csv}")
    print(df.to_string())

if __name__ == "__main__":
    main()
