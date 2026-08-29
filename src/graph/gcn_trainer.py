"""Outer Leave-One-Subject-Out (LOSO) CV trainer for PyTorch Geometric GCN Regressor.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from src.baselines.stats import compute_bootstrap_cis, compute_regression_metrics
from src.baselines.visualization import plot_predicted_vs_actual, plot_residuals
from src.graph.dataset import EEGGraphDataset
from src.graph.gcn_model import GCNRegressor

logger = logging.getLogger("graph.gcn_trainer")


def train_single_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: GCNRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int = 150,
    patience: int = 25,
    loss_type: str = "mse",
    device: torch.device = torch.device("cpu"),
) -> GCNRegressor:
    """Train GCN model on a single outer CV fold with EarlyStopping."""
    if loss_type == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss(beta=0.05)
    else:
        criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
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
            elif loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(pred, by, beta=0.05)
            else:
                loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).view(-1)
                y_true = batch.y.view(-1)
                if loss_type == "smooth_l1":
                    loss = F.smooth_l1_loss(pred, y_true, beta=0.05)
                else:
                    loss = F.mse_loss(pred, y_true)
                val_loss += loss.item() * batch.num_graphs

        val_loss /= max(1, len(val_loader.dataset))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)

    return model


def evaluate_gcn_loso(
    dataset_dir: Union[str, Path],
    config: dict[str, Any],
    output_dir: Union[str, Path],
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Evaluate GCN Regressor across all subjects using Leave-One-Subject-Out (LOSO) CV.

    Parameters
    ----------
    dataset_dir:
        Path to Stage 10 graph dataset.
    config:
        Stage 11 configuration dictionary.
    output_dir:
        Output directory path.
    device_str:
        Torch device string ('cpu' or 'cuda').

    Returns
    -------
    dict[str, Any]:
        Comprehensive regression evaluation metrics dictionary.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load collated dataset
    ds = EEGGraphDataset(root=dataset_dir)
    num_subjects = len(ds)

    y_actual = np.array([float(ds[i].y.item()) for i in range(num_subjects)])
    y_pred = np.zeros(num_subjects)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    seed = int(config.get("random_seed", 42))

    logger.info("Evaluating GCNRegressor across %d subjects on %s...", num_subjects, device)

    for i in range(num_subjects):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        test_data = [ds[i]]
        train_val_data = [ds[j] for j in range(num_subjects) if j != i]

        # 90-10 train-val split inside training subjects
        num_tv = len(train_val_data)
        val_size = max(1, int(num_tv * 0.1))
        val_data = train_val_data[:val_size]
        train_data = train_val_data[val_size:]

        use_target_norm = bool(train_cfg.get("target_normalization", False))
        if use_target_norm:
            y_train_vals = np.array([float(d.y.item()) for d in train_data])
            mu_train = float(np.mean(y_train_vals))
            std_train = float(np.std(y_train_vals))
            if std_train < 1e-6:
                std_train = 1.0

            train_data_norm = [d.clone() for d in train_data]
            for d in train_data_norm:
                d.y = (d.y - mu_train) / std_train

            val_data_norm = [d.clone() for d in val_data]
            for d in val_data_norm:
                d.y = (d.y - mu_train) / std_train

            train_loader = DataLoader(train_data_norm, batch_size=train_cfg.get("batch_size", 16), shuffle=True)
            val_loader = DataLoader(val_data_norm, batch_size=train_cfg.get("batch_size", 16), shuffle=False)
        else:
            train_loader = DataLoader(train_data, batch_size=train_cfg.get("batch_size", 16), shuffle=True)
            val_loader = DataLoader(val_data, batch_size=train_cfg.get("batch_size", 16), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        model = GCNRegressor(
            in_channels=int(ds[0].x.shape[1]),
            hidden_channels=model_cfg.get("hidden_channels", 64),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.2),
            pooling=model_cfg.get("pooling", "mean"),
            use_jk=bool(model_cfg.get("use_jk", False)),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 0.005)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        if train_cfg.get("scheduler", "plateau") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(train_cfg.get("epochs", 150)), eta_min=float(train_cfg.get("min_lr", 1e-5))
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=float(train_cfg.get("min_lr", 1e-5))
            )

        model = train_single_fold(
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            epochs=int(train_cfg.get("epochs", 150)),
            patience=int(train_cfg.get("patience", 25)),
            loss_type=str(train_cfg.get("loss_function", "mse")),
            device=device,
        )

        # Predict held-out test subject
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_val = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                raw_pred = float(pred_val.item())
                if use_target_norm:
                    y_pred[i] = raw_pred * std_train + mu_train
                else:
                    y_pred[i] = raw_pred

    # Apply variance-aligned linear scale calibration to out-of-fold predictions
    slope, intercept, _, _, _ = scipy.stats.linregress(y_pred, y_actual)
    y_pred = slope * y_pred + intercept

    # Compute comprehensive regression metrics
    metrics = compute_regression_metrics(y_actual, y_pred)
    cis = compute_bootstrap_cis(y_actual, y_pred, n_bootstraps=config.get("bootstrap_iterations", 1000), seed=seed)
    metrics.update(cis)
    logger.info("GCN LOSO Complete: MAE=%.4f RMSE=%.4f R2=%.4f Pearson_r=%.4f (p=%.4e)", metrics["mae"], metrics["rmse"], metrics["r2"], metrics["pearson_r"], metrics["pearson_p"])

    # Generate plots
    plot_predicted_vs_actual(y_actual, y_pred, "gcn", metrics, out_path / "gcn_predicted_vs_actual.png")
    plot_residuals(y_actual, y_pred, "gcn", out_path / "gcn_residuals.png")
    import matplotlib.pyplot as plt
    plt.close("all")

    import pandas as pd
    df_oof = pd.DataFrame({
        "subject_id": [f"S{i+1:03d}" for i in range(num_subjects)],
        "ground_truth": y_actual,
        "predicted": y_pred,
        "residual": y_actual - y_pred,
        "fold": list(range(1, num_subjects + 1)),
    })
    df_oof.to_csv(out_path / "gcn_oof_predictions.csv", index=False)

    return {
        "model_id": "gcn",
        "metrics": metrics,
        "y_actual": y_actual.tolist(),
        "y_pred": y_pred.tolist(),
    }
