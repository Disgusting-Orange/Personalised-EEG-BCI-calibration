"""Model registry and hyperparameter search grids for Stage 8 Baseline Regressors.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


DEFAULT_HPO_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ridge": {
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "lasso": {
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "elasticnet": {
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "svr": {
        "regressor__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "regressor__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "regressor__epsilon": [0.001, 0.01, 0.1],
    },
    "rf": {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [3, 5, 8, None],
        "regressor__min_samples_leaf": [1, 3, 5],
        "regressor__max_features": ["sqrt", "log2", 0.3],
    },
    "xgboost": {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [2, 3, 5],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__subsample": [0.7, 1.0],
        "regressor__colsample_bytree": [0.7, 1.0],
    },
}


def create_pipeline(model_id: str, random_seed: int = 42) -> Pipeline:
    """Create Scikit-Learn Pipeline with StandardScaler and specified regressor.

    Parameters
    ----------
    model_id:
        Identifier string ('dummy', 'ridge', 'lasso', 'elasticnet', 'svr', 'rf', 'xgboost').
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    Pipeline:
        Unfitted Scikit-Learn pipeline.
    """
    model_id = model_id.lower()

    if model_id == "dummy":
        reg = DummyRegressor(strategy="mean")
    elif model_id == "ridge":
        reg = Ridge(random_state=random_seed)
    elif model_id == "lasso":
        reg = Lasso(random_state=random_seed, max_iter=5000)
    elif model_id == "elasticnet":
        reg = ElasticNet(random_state=random_seed, max_iter=5000)
    elif model_id == "svr":
        reg = SVR()
    elif model_id == "rf":
        reg = RandomForestRegressor(random_state=random_seed, n_jobs=1)
    elif model_id == "xgboost":
        reg = XGBRegressor(random_state=random_seed, n_jobs=1, verbosity=0)
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Supported: dummy, ridge, lasso, elasticnet, svr, rf, xgboost")

    return Pipeline([("scaler", StandardScaler()), ("regressor", reg)])


def get_hpo_grid(model_id: str, custom_grids: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    """Get HPO search grid for model_id with parameter names prefixed for Pipeline.

    Parameters
    ----------
    model_id:
        Model identifier.
    custom_grids:
        Optional custom dictionary overriding default parameter grids.

    Returns
    -------
    dict:
        Parameter grid dictionary.
    """
    model_id = model_id.lower()
    if model_id == "dummy":
        return {}

    raw_grid = DEFAULT_HPO_GRIDS.get(model_id, {})
    if custom_grids and model_id in custom_grids:
        raw = custom_grids[model_id]
        raw_grid = {f"regressor__{k}" if not k.startswith("regressor__") else k: v for k, v in raw.items()}

    return raw_grid
