"""Outer Leave-One-Subject-Out (LOSO) Nested-CV Evaluator for Stage 8.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut

from src.baselines.models import create_pipeline, get_hpo_grid
from src.baselines.stats import compute_bootstrap_cis, compute_regression_metrics

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("baselines.evaluator")


def load_or_create_loso_splits(
    n_subjects: int,
    splits_file: Path | str | None = None,
) -> list[dict[str, list[int]]]:
    """Load or deterministically generate outer LOSO fold splits.

    Parameters
    ----------
    n_subjects:
        Number of total subjects in cohort (e.g. 109).
    splits_file:
        Path to JSON file storing LOSO split index structures.

    Returns
    -------
    list[dict[str, list[int]]]:
        List of 109 dictionaries containing 'train_idx' and 'test_idx'.
    """
    if splits_file:
        splits_path = Path(splits_file)
        if splits_path.exists():
            with splits_path.open("r", encoding="utf-8") as fh:
                splits = json.load(fh)
                if len(splits) == n_subjects:
                    return splits

    loo = LeaveOneOut()
    splits = []
    indices = np.arange(n_subjects)
    for train_idx, test_idx in loo.split(indices):
        splits.append({
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        })

    if splits_file:
        splits_path = Path(splits_file)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    return splits


def evaluate_model_loso(
    model_id: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    subject_ids: Sequence[str],
    config: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Evaluate baseline regressor under Leave-One-Subject-Out (LOSO) nested CV.

    Parameters
    ----------
    model_id:
        Regressor model identifier ('dummy', 'ridge', 'lasso', 'elasticnet', 'svr', 'rf', 'xgboost').
    X:
        Feature matrix (n_subjects, n_features).
    y:
        Continuous target vector (n_subjects,).
    feature_names:
        List of feature column names.
    subject_ids:
        List of subject ID strings.
    config:
        Stage 8 configuration dictionary.
    n_jobs:
        Parallel process workers.

    Returns
    -------
    dict[str, Any]:
        Detailed evaluation output dictionary containing OOF predictions, metrics, CIs,
        and feature importance analysis.
    """
    start_time = time.perf_counter()
    n_subjects, n_features = X.shape
    seed = int(config.get("random_seed", 42)) if config else 42
    inner_folds = int(config.get("inner_cv_folds", 5)) if config else 5
    splits_file = config.get("loso_splits_path") if config else None

    splits = load_or_create_loso_splits(n_subjects, splits_file)

    oof_predictions = np.zeros(n_subjects, dtype=np.float64)
    best_params_per_fold: list[dict[str, Any]] = []
    feature_importances_accum = np.zeros(n_features, dtype=np.float64)

    grid = get_hpo_grid(model_id, config.get("hpo_grids") if config else None)

    logger.info("Evaluating model '%s' across %d LOSO outer folds (inner_folds=%d, n_jobs=%d)...", model_id, n_subjects, inner_folds, n_jobs)

    for fold_idx, split_dict in enumerate(splits):
        train_idx = np.array(split_dict["train_idx"])
        test_idx = np.array(split_dict["test_idx"])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        pipeline = create_pipeline(model_id, random_seed=seed + fold_idx)

        if model_id == "dummy" or not grid:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
        else:
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold_idx)
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

        pred = best_model.predict(X_test)
        oof_predictions[test_idx[0]] = float(pred[0])
        best_params_per_fold.append(best_params)

        # Feature Importance Collection
        reg = best_model.named_steps["regressor"]
        if hasattr(reg, "coef_"):
            coef = np.abs(reg.coef_)
            if coef.ndim > 1:
                coef = coef.flatten()
            feature_importances_accum += coef
        elif hasattr(reg, "feature_importances_"):
            feature_importances_accum += reg.feature_importances_

    mean_feature_importance = feature_importances_accum / n_subjects

    # Metrics & Bootstrap CIs
    metrics = compute_regression_metrics(y, oof_predictions)
    n_bootstraps = int(config.get("bootstrap_iterations", 1000)) if config else 1000
    cis = compute_bootstrap_cis(y, oof_predictions, n_bootstraps=n_bootstraps, seed=seed)
    metrics.update(cis)

    # Feature Importance DataFrame & Rankings
    df_importance = pd.DataFrame({
        "feature": list(feature_names),
        "importance": mean_feature_importance,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    elapsed = time.perf_counter() - start_time
    logger.info("Model '%s' LOSO Complete in %.2fs: MAE=%.4f RMSE=%.4f R2=%.4f Pearson_r=%.4f (p=%.4e)",
                model_id, elapsed, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["pearson_r"], metrics["pearson_p"])

    return {
        "model_id": model_id,
        "n_subjects": n_subjects,
        "n_features": n_features,
        "oof_predictions": oof_predictions.tolist(),
        "ground_truth": y.tolist(),
        "subject_ids": list(subject_ids),
        "metrics": metrics,
        "best_params_sample": best_params_per_fold[0] if best_params_per_fold else {},
        "feature_importance": df_importance.to_dict(orient="records"),
        "elapsed_seconds": elapsed,
    }
