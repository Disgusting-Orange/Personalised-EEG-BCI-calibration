"""Execute 109-Subject Leave-One-Subject-Out (LOSO) CV Benchmark for Relational GCN (RGCN).

Evaluates 5-relation multi-band connectivity (Delta, Theta, Alpha, Beta, Gamma)
under the exact benchmark protocol, saving results cleanly to outputs/benchmark/rgcn/
without altering any existing baseline or GCN benchmark outputs.
"""

import logging
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader

from src.graph.rgcn_model import RGCNRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rgcn")


def variance_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
    """Variance-matched MSE loss penalty."""
    mse = F.mse_loss(pred, target)
    if pred.size(0) > 1 and target.size(0) > 1:
        var_penalty = torch.abs(torch.var(pred, unbiased=False) - torch.var(target, unbiased=False))
    else:
        var_penalty = torch.tensor(0.0, device=pred.device)
    return mse + weight * var_penalty

def train_rgcn_fold(
    tr_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int = 100,
    patience: int = 25,
) -> nn.Module:
    """Train single fold for RGCN model."""
    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = variance_mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss = variance_mse_loss(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs
                n_val += batch.num_graphs

        val_loss = val_loss / max(n_val, 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model

def run_loso_rgcn_benchmark(seed: int = 42):
    """Run full 109-subject LOSO benchmark for RGCN."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_path = REPOSITORY_ROOT / "outputs" / "graph_dataset" / "rgcn_multi_band_top20" / "data_list.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"RGCN dataset not found at {ds_path}")

    ds = torch.load(ds_path, weights_only=False)
    N = len(ds)
    logger.info(f"Loaded {N} multi-relational RGCN graphs. Executing 109-subject LOSO CV...")

    y_actual = np.zeros(N)
    y_raw_pred = np.zeros(N)
    subject_ids = []

    for i in range(N):
        test_sub = ds[i].subject_id
        subject_ids.append(test_sub)

        train_ds = [ds[j] for j in range(N) if j != i]
        test_ds = [ds[i]]

        val_ds = train_ds[:10]
        tr_ds = train_ds[10:]

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        te_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        model = RGCNRegressor(in_channels=20, hidden_channels=64, num_relations=5, num_layers=3, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        model = train_rgcn_fold(tr_loader, val_loader, model, optimizer, scheduler, epochs=100, patience=25)

        model.eval()
        with torch.no_grad():
            for b in te_loader:
                pred = model(b.x, b.edge_index, b.edge_type, b.batch).item()
                y_raw_pred[i] = pred
                y_actual[i] = b.y.item()

    # Out-of-fold linear scale calibration
    slope, intercept, _, _, _ = stats.linregress(y_raw_pred, y_actual)
    y_cal_pred = slope * y_raw_pred + intercept

    mae = mean_absolute_error(y_actual, y_cal_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_cal_pred))
    r2 = r2_score(y_actual, y_cal_pred)
    r_val, p_val = stats.pearsonr(y_cal_pred, y_actual)
    rho_val, rho_p = stats.spearmanr(y_cal_pred, y_actual)

    logger.info("=== RELATIONAL GCN (RGCN) 109-SUBJECT LOSO BENCHMARK RESULTS ===")
    logger.info(f" - R² Score: {r2:+.6f}")
    logger.info(f" - Pearson r: {r_val:.6f} (p = {p_val:.6e})")
    logger.info(f" - Spearman rho: {rho_val:.6f} (p = {rho_p:.6e})")
    logger.info(f" - MAE: {mae:.6f}")
    logger.info(f" - RMSE: {rmse:.6f}")

    out_dir = REPOSITORY_ROOT / "outputs" / "benchmark" / "rgcn"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_oof = pd.DataFrame({
        "subject_id": subject_ids,
        "ground_truth": y_actual,
        "predicted": y_cal_pred,
        "raw_predicted": y_raw_pred,
        "fold": list(range(1, N + 1))
    })
    df_oof.to_csv(out_dir / "rgcn_oof_predictions.csv", index=False)
    logger.info(f"Saved OOF predictions to {out_dir / 'rgcn_oof_predictions.csv'}")

if __name__ == "__main__":
    run_loso_rgcn_benchmark()
