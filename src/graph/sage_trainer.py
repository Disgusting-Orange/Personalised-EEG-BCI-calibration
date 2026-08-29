"""Outer Leave-One-Subject-Out (LOSO) CV trainer for PyTorch Geometric GraphSAGE Regressor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.baselines.stats import compute_bootstrap_cis, compute_regression_metrics
from src.baselines.visualization import plot_predicted_vs_actual, plot_residuals
from src.graph.dataset import EEGGraphDataset
from src.graph.sage_model import SAGERegressor

logger = logging.getLogger("graph.sage_trainer")


def train_single_sage_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SAGERegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int = 150,
    patience: int = 25,
    loss_type: str = "mse",
    device: torch.device = torch.device("cpu"),
) -> SAGERegressor:
    """Train GraphSAGE model on a single outer CV fold with EarlyStopping."""
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
            pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(pred, batch.y)
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
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = criterion(pred, batch.y)
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


def evaluate_graphsage_loso(
    dataset_dir: Union[str, Path],
    config: dict[str, Any],
    output_dir: Union[str, Path],
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Evaluate GraphSAGE Regressor across all subjects using Leave-One-Subject-Out (LOSO) CV."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = EEGGraphDataset(root=dataset_dir)
    num_subjects = len(ds)

    y_actual = np.array([float(ds[i].y.item()) for i in range(num_subjects)])
    y_pred = np.zeros(num_subjects)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    seed = int(config.get("random_seed", 42))

    logger.info("Evaluating SAGERegressor across %d subjects on %s...", num_subjects, device)

    for i in range(num_subjects):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        test_data = [ds[i]]
        train_val_data = [ds[j] for j in range(num_subjects) if j != i]

        num_tv = len(train_val_data)
        val_size = max(1, int(num_tv * 0.1))
        val_data = train_val_data[:val_size]
        train_data = train_val_data[val_size:]

        train_loader = DataLoader(train_data, batch_size=train_cfg.get("batch_size", 16), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_cfg.get("batch_size", 16), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        model = SAGERegressor(
            in_channels=model_cfg.get("in_channels", 10),
            hidden_channels=model_cfg.get("hidden_channels", 48),
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

        model = train_single_sage_fold(
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

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_val = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                y_pred[i] = float(pred_val.item())

    metrics = compute_regression_metrics(y_actual, y_pred)
    cis = compute_bootstrap_cis(y_actual, y_pred, n_bootstraps=config.get("bootstrap_iterations", 1000), seed=seed)
    metrics.update(cis)
    logger.info("GraphSAGE LOSO Complete: MAE=%.4f RMSE=%.4f R2=%.4f Pearson_r=%.4f (p=%.4e)", metrics["mae"], metrics["rmse"], metrics["r2"], metrics["pearson_r"], metrics["pearson_p"])

    plot_predicted_vs_actual(y_actual, y_pred, "graphsage", metrics, out_path / "graphsage_predicted_vs_actual.png")
    plot_residuals(y_actual, y_pred, "graphsage", out_path / "graphsage_residuals.png")

    import matplotlib.pyplot as plt
    plt.close("all")

    df_oof = pd.DataFrame({
        "subject_id": [f"S{i+1:03d}" for i in range(num_subjects)],
        "ground_truth": y_actual,
        "predicted": y_pred,
        "residual": y_actual - y_pred,
        "fold": list(range(1, num_subjects + 1)),
    })
    df_oof.to_csv(out_path / "graphsage_oof_predictions.csv", index=False)

    return {
        "model_id": "graphsage",
        "metrics": metrics,
        "y_actual": y_actual.tolist(),
        "y_pred": y_pred.tolist(),
    }
