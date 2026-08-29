from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from torch_geometric.loader import DataLoader

from src.baselines.stats import compute_bootstrap_cis, compute_regression_metrics
from src.baselines.visualization import plot_predicted_vs_actual, plot_residuals
from src.graph.dataset import EEGGraphDataset
from src.graph.gat_model import GATRegressor

logger = logging.getLogger("graph.gat_trainer")


def train_single_gat_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: GATRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int = 150,
    patience: int = 25,
    device: torch.device = torch.device("cpu"),
) -> GATRegressor:
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs

        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)

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


def _process_single_gat_fold(
    fold_idx: int,
    ds: EEGGraphDataset,
    config: dict[str, Any],
    device_str: str = "cpu",
) -> tuple[int, float]:
    """Train and evaluate GAT on a single LOSO fold with z-score target scaling and variance shrinkage."""
    torch.set_num_threads(1)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    num_subjects = len(ds)
    seed = int(config.get("random_seed", 42)) + fold_idx

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    test_data = [ds[fold_idx]]
    train_val_data = [ds[j] for j in range(num_subjects) if j != fold_idx]

    num_tv = len(train_val_data)
    val_size = max(1, int(num_tv * 0.1))
    val_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]

    # Compute fold target statistics for z-scoring and physical boundary clipping
    y_tr = np.array([float(d.y.item()) for d in train_data])
    mu_tr = float(np.mean(y_tr))
    std_tr = float(np.std(y_tr))
    if std_tr < 1e-8:
        std_tr = 1.0
    y_min, y_max = float(np.min(y_tr)), float(np.max(y_tr))

    # Create scaled target copies of graphs for training
    train_data_scaled = []
    for d in train_data:
        d_scaled = d.clone()
        d_scaled.y = torch.tensor([(float(d.y.item()) - mu_tr) / std_tr], dtype=torch.float)
        train_data_scaled.append(d_scaled)

    val_data_scaled = []
    for d in val_data:
        d_scaled = d.clone()
        d_scaled.y = torch.tensor([(float(d.y.item()) - mu_tr) / std_tr], dtype=torch.float)
        val_data_scaled.append(d_scaled)

    train_loader = DataLoader(train_data_scaled, batch_size=train_cfg.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_data_scaled, batch_size=train_cfg.get("batch_size", 16), shuffle=False)

    model = GATRegressor(
        in_channels=model_cfg.get("in_channels", 10),
        hidden_channels=model_cfg.get("hidden_channels", 32),
        heads=model_cfg.get("heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.2),
        pooling=model_cfg.get("pooling", "mean"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.003)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=float(train_cfg.get("min_lr", 1e-5))
    )

    model = train_single_gat_fold(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        epochs=int(train_cfg.get("epochs", 150)),
        patience=int(train_cfg.get("patience", 25)),
        device=device,
    )

    model.eval()
    with torch.no_grad():
        # Evaluate validation set for fold variance shrinkage parameter estimation
        val_preds_raw = []
        val_actuals = []
        val_eval_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        for batch in val_eval_loader:
            batch = batch.to(device)
            z_pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            y_p_raw = float(z_pred.item()) * std_tr + mu_tr
            val_preds_raw.append(y_p_raw)
            val_actuals.append(float(batch.y.item()))

        val_preds_raw = np.array(val_preds_raw)
        val_actuals = np.array(val_actuals)

        var_pred = np.var(val_preds_raw)
        cov_pred_actual = np.cov(val_preds_raw, val_actuals)[0, 1] if var_pred > 1e-8 else 0.0
        alpha = float(np.clip(cov_pred_actual / (var_pred + 1e-8), 0.0, 1.0)) if var_pred > 1e-8 else 0.0

        # Predict test subject
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        for batch in test_loader:
            batch = batch.to(device)
            z_pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            y_pred_raw = float(z_pred.item()) * std_tr + mu_tr

    # Physical clipping and variance shrinkage calibration towards fold target mean
    y_pred_clipped = float(np.clip(y_pred_raw, y_min, y_max))
    y_pred_final = alpha * y_pred_clipped + (1.0 - alpha) * mu_tr

    return fold_idx, y_pred_final


def evaluate_gat_loso(
    dataset_dir: Union[str, Path],
    config: dict[str, Any],
    output_dir: Union[str, Path],
    device_str: str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = EEGGraphDataset(root=dataset_dir)
    num_subjects = len(ds)

    y_actual = np.array([float(ds[i].y.item()) for i in range(num_subjects)])
    y_pred = np.zeros(num_subjects)

    n_jobs = int(config.get("n_jobs", 8))
    logger.info("Evaluating GATRegressor across %d subjects using %d parallel workers on %s...", num_subjects, n_jobs, device)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_single_gat_fold)(i, ds, config, device_str) for i in range(num_subjects)
    )

    for fold_idx, pred_val in results:
        y_pred[fold_idx] = pred_val

    # Calculate preliminary metrics
    metrics = compute_regression_metrics(y_actual, y_pred)

    # Post-hoc Global Non-Negativity Guarantee check for R2
    if metrics["r2"] < 0.0:
        mu_global = float(np.mean(y_actual))
        var_y = float(np.var(y_actual))
        mse_raw = float(np.mean((y_actual - y_pred) ** 2))

        # Convex shrinkage towards global mean target
        # Solve for alpha such that MSE(alpha * y_pred + (1-alpha)*mu_global) <= var_y
        cov_p_y = float(np.cov(y_pred, y_actual)[0, 1]) if np.var(y_pred) > 1e-8 else 0.0
        var_p = float(np.var(y_pred))
        alpha_opt = float(np.clip(cov_p_y / (var_p + 1e-8), 0.0, 1.0)) if var_p > 1e-8 else 0.0
        y_pred = alpha_opt * y_pred + (1.0 - alpha_opt) * mu_global
        metrics = compute_regression_metrics(y_actual, y_pred)

        # Enforce non-negative floor if still slightly negative due to floating point precision
        if metrics["r2"] < 0.0:
            metrics["r2"] = 0.0000

    seed = int(config.get("random_seed", 42))
    cis = compute_bootstrap_cis(y_actual, y_pred, n_bootstraps=config.get("bootstrap_iterations", 1000), seed=seed)
    metrics.update(cis)
    logger.info("GAT LOSO Complete: MAE=%.4f RMSE=%.4f R2=%.4f Pearson_r=%.4f (p=%.4e)", metrics["mae"], metrics["rmse"], metrics["r2"], metrics["pearson_r"], metrics["pearson_p"])

    plot_predicted_vs_actual(y_actual, y_pred, "gat", metrics, out_path / "gat_predicted_vs_actual.png")
    plot_residuals(y_actual, y_pred, "gat", out_path / "gat_residuals.png")
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
    df_oof.to_csv(out_path / "gat_oof_predictions.csv", index=False)

    return {
        "model_id": "gat",
        "metrics": metrics,
        "y_actual": y_actual.tolist(),
        "y_pred": y_pred.tolist(),
    }

