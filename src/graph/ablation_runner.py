"""Publication-Quality Scientific Ablation Suite Engine for Stage 16.

Evaluates 7 distinct scientific ablation studies isolating one component at a time
using the frozen primary GCN regressor under 109-subject LOSO cross-validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch_geometric.loader import DataLoader

from src.graph.builder import build_subject_graph
from src.graph.dataset import EEGGraphDataset
from src.graph.gcn_model import GCNRegressor
from src.graph.gcn_trainer import train_single_fold

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("graph.ablation_runner")


def load_subject_ids(splits_path: Union[str, Path] | None = None) -> List[str]:
    """Load ordered 109 subject IDs."""
    if splits_path:
        path = Path(splits_path)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [f"S{idx+1:03d}" for idx in range(len(data))]
                elif isinstance(data, dict) and "subject_ids" in data:
                    return data["subject_ids"]

    # Fallback to scanning targets
    targets_dir = REPOSITORY_ROOT / "outputs" / "targets"
    sub_dirs = sorted([d for d in targets_dir.glob("stage3_s*") if d.is_dir()])
    sub_ids = []
    for d in sub_dirs:
        sub_str = d.name.replace("stage3_", "").upper()
        sub_ids.append(sub_str)
    return sorted(sub_ids)


def get_or_build_graph_dataset(
    subject_ids: List[str],
    conn_metric: str = "wpli",
    freq_band: str = "alpha",
    density: float = 0.20,
) -> Any:
    """Dynamically build or load EEGGraphDataset for specified subjects under given config."""
    top_int = int(round(density * 100))
    prebuilt_dir = REPOSITORY_ROOT / "outputs" / "graph_dataset" / f"{conn_metric}_{freq_band}_top{top_int}"
    if prebuilt_dir.exists() and (prebuilt_dir / "pyg_dataset.pt").exists():
        ds = EEGGraphDataset(root=prebuilt_dir)
        if len(subject_ids) < len(ds):
            sub_set = set(subject_ids)
            indices = [i for i in range(len(ds)) if f"S{i+1:03d}" in sub_set or getattr(ds[i], "subject_id", [f"S{i+1:03d}"])[0] in sub_set]
            return ds[indices]
        return ds

    cfg = {
        "connectivity_metric": conn_metric,
        "frequency_band": freq_band,
        "sparsification_density": density,
        "node_feature_type": "concatenated",
        "ensure_connected": True,
    }
    return EEGGraphDataset(root=prebuilt_dir, subjects=subject_ids, config=cfg)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-12)))

    if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r_pearson, p_pearson = stats.pearsonr(y_true, y_pred)
        r_spearman, p_spearman = stats.spearmanr(y_true, y_pred)
    else:
        r_pearson, p_pearson = 0.0, 1.0
        r_spearman, p_spearman = 0.0, 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
    }


from src.baselines.stats import compute_regression_metrics
from joblib import Parallel, delayed


def evaluate_loso_ablation(
    dataset: List[Any],
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    seed: int = 42,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate GCN under 109-subject LOSO deterministically for a specific ablation configuration."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    n_subjects = len(dataset)
    y_true_all = np.array([float(dataset[i].y.item()) for i in range(n_subjects)])
    y_pred_all = np.zeros(n_subjects)
    subject_ids_all = []

    for idx in range(n_subjects):
        torch.manual_seed(seed + idx)
        np.random.seed(seed + idx)

        test_data = [dataset[idx]]
        train_val_data = [dataset[i] for i in range(n_subjects) if i != idx]

        # 90-10 train-val split inside training subjects matching Stage 11 evaluate_gcn_loso
        n_tv = len(train_val_data)
        n_val = max(1, int(n_tv * 0.1))
        val_data = train_val_data[:n_val]
        train_data = train_val_data[n_val:]

        batch_size = train_kwargs.get("batch_size", 16)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        model = GCNRegressor(
            in_channels=model_kwargs.get("in_channels", 10),
            hidden_channels=model_kwargs.get("hidden_channels", 64),
            num_layers=model_kwargs.get("num_layers", 3),
            dropout=model_kwargs.get("dropout", 0.2),
            pooling=model_kwargs.get("pooling", "concat"),
            use_jk=bool(model_kwargs.get("use_jk", True)),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_kwargs.get("lr", 0.005)),
            weight_decay=float(train_kwargs.get("weight_decay", 1e-4)),
        )

        epochs = int(train_kwargs.get("epochs", 150))
        min_lr = float(train_kwargs.get("min_lr", 1e-5))
        scheduler_type = train_kwargs.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=min_lr
            )

        model = train_single_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            patience=int(train_kwargs.get("patience", 25)),
            loss_type=str(train_kwargs.get("loss_function", "smooth_l1")),
            device=device,
        )

        model.eval()
        with torch.no_grad():
            for b in test_loader:
                b = b.to(device)
                pred = model(b.x, b.edge_index, b.edge_weight, b.batch)
                y_pred_all[idx] = float(pred.item())
                subject_ids_all.append(b.subject_id[0] if hasattr(b, "subject_id") else f"S{idx+1:03d}")

    metrics = compute_regression_metrics(y_true_all, y_pred_all)
    metrics["subject_ids"] = subject_ids_all
    metrics["y_true"] = y_true_all.tolist()
    metrics["y_pred"] = y_pred_all.tolist()
    metrics["errors"] = np.abs(y_true_all - y_pred_all).tolist()

    return metrics


def run_all_ablations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run all 7 scientific ablation experiments and compute statistical comparisons."""
    out_dir = Path(config.get("output_directory", "outputs/ablation_studies"))
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = load_subject_ids(config.get("loso_splits_path"))
    logger.info(f"Loaded {len(subject_ids)} subject IDs for Stage 16 Ablations.")

    base_model_cfg = dict(config["baseline_model"])
    base_train_cfg = dict(config["baseline_training"])
    base_train_cfg["n_jobs"] = int(config.get("n_jobs", 8))

    conn_metric_base = base_model_cfg.pop("connectivity_metric", "wpli")
    freq_band_base = base_model_cfg.pop("frequency_band", "alpha")
    density_base = base_model_cfg.pop("sparsification_density", 0.20)

    results: Dict[str, Any] = {}
    csv_rows = []

    device_str = config.get("device", "cpu")

    # Pre-build baseline graph dataset
    logger.info(f"Pre-building baseline graph dataset ({conn_metric_base}, {freq_band_base}, top{int(density_base*100)})...")
    baseline_dataset = get_or_build_graph_dataset(subject_ids, conn_metric_base, freq_band_base, density_base)

    # 0. Run Baseline Model once and store as baseline result reference
    logger.info("Running Baseline Frozen Primary GCN Model...")
    baseline_res = evaluate_loso_ablation(baseline_dataset, base_model_cfg, base_train_cfg, device_str=device_str)

    def run_experiment(exp_name: str, group_name: str, ds: List[Any], m_kwargs: Dict[str, Any], t_kwargs: Dict[str, Any], is_base_variant: bool = False):
        if is_base_variant:
            logger.info(f"Ablation: [{exp_name}] -> {group_name} matches baseline. Reusing baseline results.")
            res = dict(baseline_res)
        else:
            logger.info(f"Running Ablation: [{exp_name}] -> {group_name}...")
            res = evaluate_loso_ablation(ds, m_kwargs, t_kwargs, device_str=device_str)
        res["exp_name"] = exp_name
        res["group_name"] = group_name
        results[f"{exp_name}_{group_name}"] = res

        csv_rows.append({
            "Experiment": exp_name,
            "Variant": group_name,
            "R2": res["r2"],
            "MAE": res["mae"],
            "RMSE": res["rmse"],
            "Pearson_r": res["pearson_r"],
            "Pearson_p": res["pearson_p"],
            "Spearman_r": res["spearman_r"],
            "Spearman_p": res["spearman_p"],
        })
        return res

    # 1. Topology Density Ablation
    exp1_defs = config["ablations"]["topology_density"]
    for dens, name in zip(exp1_defs["densities"], exp1_defs["names"]):
        if dens == density_base:
            ds = baseline_dataset
        else:
            ds = get_or_build_graph_dataset(subject_ids, conn_metric_base, freq_band_base, dens)
        run_experiment("topology_density", name, ds, base_model_cfg, base_train_cfg, is_base_variant=(name == exp1_defs["baseline"]))

    # 2. Connectivity Metric Ablation
    exp2_defs = config["ablations"]["connectivity_metric"]
    for metric, name in zip(exp2_defs["metrics"], exp2_defs["names"]):
        if metric == conn_metric_base:
            ds = baseline_dataset
        else:
            ds = get_or_build_graph_dataset(subject_ids, metric, freq_band_base, density_base)
        run_experiment("connectivity_metric", name, ds, base_model_cfg, base_train_cfg, is_base_variant=(name == exp2_defs["baseline"]))

    # 3. Frequency Band Ablation
    exp3_defs = config["ablations"]["frequency_band"]
    for band, name in zip(exp3_defs["bands"], exp3_defs["names"]):
        if band == freq_band_base:
            ds = baseline_dataset
        else:
            ds = get_or_build_graph_dataset(subject_ids, conn_metric_base, band, density_base)
        run_experiment("frequency_band", name, ds, base_model_cfg, base_train_cfg, is_base_variant=(name == exp3_defs["baseline"]))

    # 4. Pooling Strategy Ablation
    exp4_defs = config["ablations"]["pooling_strategy"]
    for pool, name in zip(exp4_defs["poolings"], exp4_defs["names"]):
        m_cfg = dict(base_model_cfg)
        m_cfg["pooling"] = pool
        run_experiment("pooling_strategy", name, baseline_dataset, m_cfg, base_train_cfg, is_base_variant=(name == exp4_defs["baseline"]))

    # 5. Loss Function Ablation
    exp5_defs = config["ablations"]["loss_function"]
    for loss_f, name in zip(exp5_defs["losses"], exp5_defs["names"]):
        t_cfg = dict(base_train_cfg)
        t_cfg["loss_function"] = loss_f
        run_experiment("loss_function", name, baseline_dataset, base_model_cfg, t_cfg, is_base_variant=(name == exp5_defs["baseline"]))

    # 6. JumpingKnowledge Ablation
    exp6_defs = config["ablations"]["jumping_knowledge"]
    for jk_opt, name in zip(exp6_defs["jk_options"], exp6_defs["names"]):
        m_cfg = dict(base_model_cfg)
        m_cfg["use_jk"] = jk_opt
        run_experiment("jumping_knowledge", name, baseline_dataset, m_cfg, base_train_cfg, is_base_variant=(name == exp6_defs["baseline"]))

    # 7. Learning Rate Scheduler Ablation
    exp7_defs = config["ablations"]["learning_rate_scheduler"]
    for sched, name in zip(exp7_defs["schedulers"], exp7_defs["names"]):
        t_cfg = dict(base_train_cfg)
        t_cfg["scheduler"] = sched
        run_experiment("learning_rate_scheduler", name, baseline_dataset, base_model_cfg, t_cfg, is_base_variant=(name == exp7_defs["baseline"]))

    # Save CSV Results
    df_results = pd.DataFrame(csv_rows)
    df_results.to_csv(out_dir / "ablation_results.csv", index=False)
    logger.info(f"Saved ablation results CSV to {out_dir / 'ablation_results.csv'}")

    # Compute Statistical Comparisons against Baseline for each study
    statistical_summary = {}
    for exp_id, exp_meta in config["ablations"].items():
        base_variant = exp_meta["baseline"]
        base_key = f"{exp_id}_{base_variant}"
        if base_key not in results:
            continue
        base_errs = np.array(results[base_key]["errors"])

        comp_list = []
        for name in exp_meta["names"]:
            curr_key = f"{exp_id}_{name}"
            if curr_key not in results:
                continue
            curr_errs = np.array(results[curr_key]["errors"])
            err_diff = curr_errs - base_errs

            if name == base_variant:
                stat, p_val, r_rb = 0.0, 1.0, 0.0
                cohen_d = 0.0
            else:
                stat, p_val = stats.wilcoxon(curr_errs, base_errs)
                # Effect size calculation
                n = len(err_diff)
                cohen_d = float(np.mean(err_diff) / max(np.std(err_diff), 1e-12))
                pos_ranks = np.sum(err_diff > 0)
                r_rb = float((2 * pos_ranks / n) - 1.0)

            comp_list.append({
                "variant": name,
                "r2": results[curr_key]["r2"],
                "mae": results[curr_key]["mae"],
                "pearson_r": results[curr_key]["pearson_r"],
                "wilcoxon_stat": float(stat),
                "p_value": float(p_val),
                "rank_biserial_r": float(r_rb),
                "cohens_dz": float(cohen_d),
            })
        statistical_summary[exp_id] = comp_list

    # Save Validation JSON Report
    report = {
        "stage": 16,
        "n_subjects": len(subject_ids),
        "baseline_performance": {
            "r2": results["topology_density_top20"]["r2"],
            "mae": results["topology_density_top20"]["mae"],
            "pearson_r": results["topology_density_top20"]["pearson_r"],
        },
        "ablations": statistical_summary,
    }

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved validation report JSON to {out_dir / 'validation_report.json'}")
    return {"results": results, "summary": statistical_summary, "df": df_results}
