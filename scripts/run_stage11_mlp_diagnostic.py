"""Standalone execution script for MLP Diagnostic Bottleneck Experiment.

Evaluates 109-subject Leave-One-Subject-Out (LOSO) CV on flattened node features
(64 nodes x 10 features = 640 dimensions) ignoring graph edges.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from baselines.mlp_model import MLPRegressor
from baselines.stats import compute_bootstrap_cis, compute_regression_metrics
from graph.dataset import EEGGraphDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("mlp_diagnostic")


def evaluate_mlp_loso(dataset_dir: Path, out_dir: Path, seed: int = 42, epochs: int = 150, lr: float = 0.005) -> dict:
    """Evaluate MLPRegressor under LOSO CV on flattened node features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = EEGGraphDataset(root=dataset_dir)
    num_subjects = len(ds)

    # Flatten node features for each subject into X_flat matrix of shape (num_subjects, 640)
    X_list = []
    y_list = []
    for i in range(num_subjects):
        x_node = ds[i].x.cpu().numpy()  # (64, 10)
        X_list.append(x_node.flatten())  # (640,)
        y_list.append(float(ds[i].y.item()))

    X_all = np.array(X_list, dtype=np.float32)  # (109, 640)
    y_actual = np.array(y_list, dtype=np.float32)  # (109,)
    y_pred = np.zeros(num_subjects, dtype=np.float32)

    logger.info("Evaluating MLPRegressor across %d subjects (Flattened Input shape: %s)...", num_subjects, X_all.shape)

    for i in range(num_subjects):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        X_test, y_test = X_all[i:i+1], y_actual[i:i+1]
        train_indices = [j for j in range(num_subjects) if j != i]
        X_tv, y_tv = X_all[train_indices], y_actual[train_indices]

        num_tv = len(train_indices)
        val_size = max(1, int(num_tv * 0.1))
        X_val, y_val = X_tv[:val_size], y_tv[:val_size]
        X_train, y_train = X_tv[val_size:], y_tv[val_size:]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        model = MLPRegressor(in_features=640, hidden_dim=128, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                p = model(bx)
                loss = criterion(p, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    p = model(bx)
                    val_loss += criterion(p, by).item() * len(by)
            val_loss /= max(1, len(val_ds))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_weights:
            model.load_state_dict(best_weights)

        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                p = model(bx)
                y_pred[i] = float(p.item())

    metrics = compute_regression_metrics(y_actual, y_pred)
    cis = compute_bootstrap_cis(y_actual, y_pred, n_bootstraps=1000, seed=seed)
    metrics.update(cis)

    df_oof = pd.DataFrame({
        "subject_id": [f"S{i+1:03d}" for i in range(num_subjects)],
        "ground_truth": y_actual,
        "predicted": y_pred,
        "residual": y_actual - y_pred,
        "fold": list(range(1, num_subjects + 1)),
    })
    df_oof.to_csv(out_dir / "mlp_oof_predictions.csv", index=False)

    return {
        "metrics": metrics,
        "y_actual": y_actual,
        "y_pred": y_pred,
        "df_oof": df_oof,
    }


def main() -> int:
    dataset_dir = REPOSITORY_ROOT / "outputs/graph_dataset/wpli_alpha_top20"
    out_dir = REPOSITORY_ROOT / "outputs/benchmark/stage11"
    start_time = time.perf_counter()

    res = evaluate_mlp_loso(dataset_dir, out_dir, seed=42)
    elapsed = time.perf_counter() - start_time

    m = res["metrics"]
    logger.info("==================================================")
    logger.info("MLP Diagnostic Complete in %.2fs", elapsed)
    logger.info("MAE: %.6f", m["mae"])
    logger.info("RMSE: %.6f", m["rmse"])
    logger.info("R2: %.6f", m["r2"])
    logger.info("Pearson r: %.6f (p=%.4e)", m["pearson_r"], m["pearson_p"])
    logger.info("Spearman rho: %.6f (p=%.4e)", m["spearman_r"], m["spearman_p"])
    logger.info("==================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
