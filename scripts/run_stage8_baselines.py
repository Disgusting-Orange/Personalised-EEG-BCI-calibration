"""Standalone execution script for Stage 8 — Classical Regression Benchmark Suite.

Performs Leave-One-Subject-Out (LOSO) nested CV evaluation for baseline regressors,
updates the append-only master results ledger, and generates 300 DPI publication figures.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from baselines.evaluator import evaluate_model_loso
from baselines.feature_loader import load_dataset
from baselines.stats import paired_model_comparison
from baselines.visualization import (
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
)


def get_git_commit() -> str:
    """Get current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def append_to_ledger(
    ledger_path: Path,
    record: dict[str, Any],
) -> None:
    """Append a single experiment record to the master CSV ledger (never overwrite)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])

    if ledger_path.exists():
        df_existing = pd.read_csv(ledger_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(ledger_path, index=False)
    else:
        df_new.to_csv(ledger_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 8 Classical Baseline Regressors Suite.")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (dummy, ridge, lasso, elasticnet, svr, rf, xgboost).")
    parser.add_argument("--feature-set", help="Feature set representation.", default="spectral_concatenated")
    parser.add_argument("--config", help="Path to Stage 8 configuration file.", default="configs/stage8_classical_baselines.yaml")
    parser.add_argument("--n-jobs", type=int, help="Parallel processing jobs.", default=4)
    parser.add_argument("--resume", action="store_true", help="Skip models with existing completion reports.")
    parser.add_argument("--subjects", nargs="+", help="Optional explicit subject IDs subset.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage8")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    feature_set = args.feature_set or cfg.get("default_feature_set", "spectral_concatenated")
    models_to_run = args.models or cfg.get("models", ["dummy", "ridge", "lasso", "elasticnet", "svr", "rf", "xgboost"])
    n_jobs = args.n_jobs or int(cfg.get("n_jobs", 4))
    ledger_path = REPOSITORY_ROOT / cfg.get("ledger_path", "reports/benchmark_ledger.csv")
    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "outputs/benchmark/stage8") / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Stage 8 dataset for feature_set='%s'...", feature_set)
    X, y, subject_ids, feature_names = load_dataset(
        feature_set=feature_set,
        subjects=args.subjects,
        targets_dir=REPOSITORY_ROOT / cfg.get("targets_dir", "outputs/targets"),
        features_dir=REPOSITORY_ROOT / cfg.get("features_dir", "outputs/features"),
        connectivity_dir=REPOSITORY_ROOT / cfg.get("connectivity_dir", "outputs/connectivity"),
    )

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary_list: list[dict[str, Any]] = []
    oof_store: dict[str, np.ndarray] = {}

    for m_id in models_to_run:
        model_out_dir = out_dir / m_id
        report_file = model_out_dir / f"{m_id}_report.json"

        if args.resume and report_file.exists():
            logger.info("Skipping model '%s' (Resume flag set and report exists).", m_id)
            with report_file.open("r", encoding="utf-8") as fh:
                rep = json.load(fh)
                summary_list.append(rep["metrics_summary"])
                oof_store[m_id] = np.array(rep["oof_predictions"])
            continue

        logger.info("==================================================")
        logger.info("Running Stage 8 Baseline Model: %s", m_id.upper())
        logger.info("==================================================")

        res = evaluate_model_loso(
            model_id=m_id,
            X=X,
            y=y,
            feature_names=feature_names,
            subject_ids=subject_ids,
            config=cfg,
            n_jobs=n_jobs,
        )

        model_out_dir.mkdir(parents=True, exist_ok=True)
        oof_pred = np.array(res["oof_predictions"])
        oof_store[m_id] = oof_pred
        metrics = res["metrics"]

        # Generate Visualizations
        plot_predicted_vs_actual(y, oof_pred, m_id, metrics, model_out_dir / f"{m_id}_predicted_vs_actual.png")
        plot_residuals(y, oof_pred, m_id, model_out_dir / f"{m_id}_residuals.png")

        df_imp = pd.DataFrame(res["feature_importance"])
        df_imp.to_csv(model_out_dir / f"{m_id}_feature_importance.csv", index=False)
        plot_feature_importance(df_imp, m_id, top_k=20, output_path=model_out_dir / f"{m_id}_feature_importance.png")

        # Save OOF predictions CSV
        df_oof = pd.DataFrame({
            "subject_id": subject_ids,
            "ground_truth": y,
            "predicted": oof_pred,
            "residual": y - oof_pred,
        })
        df_oof.to_csv(model_out_dir / f"{m_id}_oof_predictions.csv", index=False)

        # Save Model JSON Report
        report_data = {
            "model_id": m_id,
            "feature_set": feature_set,
            "git_commit": git_commit,
            "timestamp": timestamp,
            "n_subjects": len(subject_ids),
            "n_features": len(feature_names),
            "metrics": metrics,
            "metrics_summary": {
                "model_id": m_id,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "pearson_r": metrics["pearson_r"],
                "pearson_p": metrics["pearson_p"],
                "mae_ci": metrics["mae_ci"],
                "rmse_ci": metrics["rmse_ci"],
                "r2_ci": metrics["r2_ci"],
                "pearson_ci": metrics["pearson_ci"],
            },
            "best_params_sample": res["best_params_sample"],
            "oof_predictions": oof_pred.tolist(),
            "elapsed_seconds": res["elapsed_seconds"],
        }
        report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        summary_list.append(report_data["metrics_summary"])

        # Append experiment to master ledger
        ledger_record = {
            "Timestamp": timestamp,
            "Git_Commit": git_commit,
            "Model": m_id,
            "Feature_Set": feature_set,
            "Seed": cfg.get("random_seed", 42),
            "Hyperparameters": json.dumps(res["best_params_sample"]),
            "CV_Strategy": "LOSO_Outer_5Fold_Inner",
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
            "Pearson_r": metrics["pearson_r"],
            "Pearson_p": metrics["pearson_p"],
            "Spearman_r": metrics["spearman_r"],
            "Spearman_p": metrics["spearman_p"],
            "Median_AE": metrics["median_ae"],
            "Explained_Variance": metrics["explained_variance"],
            "Runtime_sec": res["elapsed_seconds"],
            "Output_Directory": str(model_out_dir),
        }
        append_to_ledger(ledger_path, ledger_record)

    # Cross-Model Comparison & Plot
    if summary_list:
        plot_model_comparison(summary_list, out_dir / "model_comparison.png")

        # Paired comparisons against Dummy baseline
        if "dummy" in oof_store:
            dummy_oof = oof_store["dummy"]
            comparison_results: dict[str, Any] = {}
            for m_id, pred_oof in oof_store.items():
                if m_id != "dummy":
                    comp = paired_model_comparison(y, pred_oof, dummy_oof)
                    comparison_results[f"{m_id}_vs_dummy"] = comp
            
            (out_dir / "paired_comparisons.json").write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    logger.info("Stage 8 complete for %d models. Master ledger updated at %s", len(models_to_run), ledger_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
