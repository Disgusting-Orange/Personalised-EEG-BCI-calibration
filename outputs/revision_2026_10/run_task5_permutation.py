"""
Task 5 — Subject-Level Target-Permutation Test.

Permutes y across subjects while holding features and graphs fixed to their subjects.
Computes null distribution of R2 and Pearson r for:
  - Hybrid Ensemble (observed R2 = +0.2221)
  - Primary GCN (observed R2 = +0.2119)
  - Random Forest (observed R2 = +0.1857)
  - Ridge Regression (observed R2 = +0.1070)

Computes empirical one-sided p-values:
  p = (1 + #{R2_perm >= R2_obs}) / (1 + N_perm)
Reports null mean, null standard deviation, and 95th percentile.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("task5_permutation")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "revision_2026_10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df_ledger = pd.read_csv(REPO_ROOT / "LOSO_OOF_predictions.csv")
    y_actual = df_ledger["actual_mi_accuracy"].values
    sub_ids = df_ledger["subject_id"].values
    
    # Load 10-D / 640-D relative spectral features
    feat_dir = REPO_ROOT / "outputs" / "features"
    relative_cols = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"]
    X_spec_list = []
    for i in range(1, 110):
        sub = f"S{i:03d}"
        sub_l = sub.lower()
        f_eo = feat_dir / f"stage6_{sub_l}" / f"{sub}_R01_band_powers.csv"
        f_ec = feat_dir / f"stage6_{sub_l}" / f"{sub}_R02_band_powers.csv"
        df_eo = pd.read_csv(f_eo, index_col=0)
        df_ec = pd.read_csv(f_ec, index_col=0)
        eo_rel = df_eo[relative_cols].values # (64, 5)
        ec_rel = df_ec[relative_cols].values # (64, 5)
        # Average across channels to get 10-D sensorimotor spectral summary
        eo_mean = np.mean(eo_rel, axis=0) # (5,)
        ec_mean = np.mean(ec_rel, axis=0) # (5,)
        X_spec_list.append(np.concatenate([eo_mean, ec_mean]))
    X_spec = np.array(X_spec_list, dtype=np.float32) # (109, 10)
    
    return sub_ids, y_actual, X_spec, df_ledger

def evaluate_rf_loso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    N = len(y)
    preds = np.zeros(N)
    for i in range(N):
        tr_idx = [j for j in range(N) if j != i]
        rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42 + i, n_jobs=1)
        rf.fit(X[tr_idx], y[tr_idx])
        preds[i] = float(rf.predict(X[[i]])[0])
    return preds

def evaluate_ridge_loso(X: np.ndarray, y: np.ndarray, alpha: float = 1000.0) -> np.ndarray:
    N = len(y)
    preds = np.zeros(N)
    for i in range(N):
        tr_idx = [j for j in range(N) if j != i]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[[i]])
        ridge = Ridge(alpha=alpha, random_state=42 + i)
        ridge.fit(X_tr, y[tr_idx])
        preds[i] = float(ridge.predict(X_te)[0])
    return preds

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9 else 0.0
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    mae = float(100.0 * np.mean(np.abs(err)))
    rmse = float(100.0 * np.sqrt(np.mean(err ** 2)))
    return {"r": r, "R2": r2, "MAE": mae, "RMSE": rmse}

def run_permutation_tests(n_perm: int = 1000, seed: int = 42):
    t_start = time.time()
    logger.info("Starting Task 5 — Subject-Level Target-Permutation Test (N_perm=%d, seed=%d)...", n_perm, seed)
    
    sub_ids, y_actual, X_spec, df_ledger = load_data()
    N = len(y_actual)
    
    # Headline observed values
    obs_preds = {
        "hybrid": df_ledger["pred_gcn_rf_hybrid"].values,
        "gcn": df_ledger["pred_base_gcn_alpha"].values,
        "rf": df_ledger["pred_rf_spectral"].values,
        "ridge": df_ledger["pred_ridge_spectral"].values,
    }
    
    obs_metrics = {k: compute_metrics(y_actual, obs_preds[k]) for k in obs_preds}
    logger.info("Observed Metrics:")
    for k, m in obs_metrics.items():
        logger.info("  %s: r=%.4f, R2=%+.4f, MAE=%.2f, RMSE=%.2f", k.upper(), m["r"], m["R2"], m["MAE"], m["RMSE"])
        
    # Numerical sanity checks on observed predictions
    for k, p in obs_preds.items():
        assert not np.any(np.isnan(p)), f"NaN detected in {k} predictions!"
        assert not np.any(np.isinf(p)), f"Inf detected in {k} predictions!"
        # Primary models strictly in [0, 1]; unconstrained Ridge within [0, 1] +- 0.10
        margin = 0.10 if k == "ridge" else 0.02
        assert np.all(p >= -margin) and np.all(p <= 1.0 + margin), f"{k} predictions outside [0, 1] +- {margin}!"
    logger.info("Sanity check passed: all observed predictions are non-NaN, finite, and within [0, 1] +- margin.")
    
    rng = np.random.default_rng(seed)
    
    # 1. Permutation on Fixed OOF Predictions (Standard chance-level reference across N=1000 permutations)
    logger.info("[1/3] Computing fixed-prediction chance reference across %d permutations for all 4 models...", n_perm)
    null_r2_fixed = {k: np.zeros(n_perm) for k in obs_preds}
    null_r_fixed = {k: np.zeros(n_perm) for k in obs_preds}
    
    for b in range(n_perm):
        perm_idx = rng.permutation(N)
        y_perm = y_actual[perm_idx]
        for k in obs_preds:
            m = compute_metrics(y_perm, obs_preds[k])
            null_r2_fixed[k][b] = m["R2"]
            null_r_fixed[k][b] = m["r"]
            
    summary_fixed = {}
    for k in obs_preds:
        obs_r2 = obs_metrics[k]["R2"]
        n_greater = int(np.sum(null_r2_fixed[k] >= obs_r2))
        p_val = float((1.0 + n_greater) / (1.0 + n_perm))
        summary_fixed[k] = {
            "model": k,
            "obs_r2": obs_r2,
            "obs_r": obs_metrics[k]["r"],
            "n_perm": n_perm,
            "n_greater_equal": n_greater,
            "empirical_p": p_val,
            "null_r2_mean": float(np.mean(null_r2_fixed[k])),
            "null_r2_sd": float(np.std(null_r2_fixed[k])),
            "null_r2_q95": float(np.percentile(null_r2_fixed[k], 95)),
            "null_r2_q99": float(np.percentile(null_r2_fixed[k], 99)),
            "null_r_mean": float(np.mean(null_r_fixed[k])),
            "null_r_sd": float(np.std(null_r_fixed[k])),
            "null_r_q95": float(np.percentile(null_r_fixed[k], 95)),
        }
        logger.info("  %s (fixed chance): p=%.4e, null R2 mean=%+.4f (q95=%+.4f), null r mean=%+.4f (q95=%.4f)",
                    k.upper(), p_val, summary_fixed[k]["null_r2_mean"], summary_fixed[k]["null_r2_q95"],
                    summary_fixed[k]["null_r_mean"], summary_fixed[k]["null_r_q95"])
                    
    # 2. Refitting Permutation Test for Ridge (1000 refits)
    logger.info("[2/3] Running full LOSO-109 refit permutation test for Ridge (N_perm=1000)...")
    n_ridge_perm = 1000
    null_r2_ridge = np.zeros(n_ridge_perm)
    null_r_ridge = np.zeros(n_ridge_perm)
    rng_ridge = np.random.default_rng(seed + 100)
    for b in range(n_ridge_perm):
        perm_idx = rng_ridge.permutation(N)
        y_perm = y_actual[perm_idx]
        pred_ridge = evaluate_ridge_loso(X_spec, y_perm, alpha=1000.0)
        m = compute_metrics(y_perm, pred_ridge)
        null_r2_ridge[b] = m["R2"]
        null_r_ridge[b] = m["r"]
    
    n_greater_ridge = int(np.sum(null_r2_ridge >= obs_metrics["ridge"]["R2"]))
    summary_refit = {
        "ridge": {
            "model": "ridge",
            "obs_r2": obs_metrics["ridge"]["R2"],
            "obs_r": obs_metrics["ridge"]["r"],
            "n_perm": n_ridge_perm,
            "n_greater_equal": n_greater_ridge,
            "empirical_p": float((1.0 + n_greater_ridge) / (1.0 + n_ridge_perm)),
            "null_r2_mean": float(np.mean(null_r2_ridge)),
            "null_r2_sd": float(np.std(null_r2_ridge)),
            "null_r2_q95": float(np.percentile(null_r2_ridge, 95)),
            "null_r_mean": float(np.mean(null_r_ridge)),
            "null_r_sd": float(np.std(null_r_ridge)),
            "null_r_q95": float(np.percentile(null_r_ridge, 95)),
        }
    }
    logger.info("  RIDGE (refit 1000): p=%.4e, null R2 mean=%+.4f (q95=%+.4f), null r mean=%+.4f (q95=%.4f)",
                summary_refit["ridge"]["empirical_p"], summary_refit["ridge"]["null_r2_mean"],
                summary_refit["ridge"]["null_r2_q95"], summary_refit["ridge"]["null_r_mean"],
                summary_refit["ridge"]["null_r_q95"])

    # 3. Refitting Permutation Test for Random Forest (N_perm=200 using parallel workers)
    logger.info("[3/3] Running full LOSO-109 refit permutation test for RF (N_perm=200)...")
    from joblib import Parallel, delayed
    n_rf_perm = 200
    rng_rf = np.random.default_rng(seed + 200)
    perm_targets = [y_actual[rng_rf.permutation(N)] for _ in range(n_rf_perm)]
    
    def refit_rf_worker(b_idx, y_p):
        preds = evaluate_rf_loso(X_spec, y_p)
        m = compute_metrics(y_p, preds)
        return m["R2"], m["r"]
        
    rf_results = Parallel(n_jobs=8)(delayed(refit_rf_worker)(b, perm_targets[b]) for b in range(n_rf_perm))
    null_r2_rf = np.array([r[0] for r in rf_results])
    null_r_rf = np.array([r[1] for r in rf_results])
    
    n_greater_rf = int(np.sum(null_r2_rf >= obs_metrics["rf"]["R2"]))
    summary_refit["rf"] = {
        "model": "rf",
        "obs_r2": obs_metrics["rf"]["R2"],
        "obs_r": obs_metrics["rf"]["r"],
        "n_perm": n_rf_perm,
        "n_greater_equal": n_greater_rf,
        "empirical_p": float((1.0 + n_greater_rf) / (1.0 + n_rf_perm)),
        "null_r2_mean": float(np.mean(null_r2_rf)),
        "null_r2_sd": float(np.std(null_r2_rf)),
        "null_r2_q95": float(np.percentile(null_r2_rf, 95)),
        "null_r_mean": float(np.mean(null_r_rf)),
        "null_r_sd": float(np.std(null_r_rf)),
        "null_r_q95": float(np.percentile(null_r_rf, 95)),
    }
    logger.info("  RF (refit 200): p=%.4e, null R2 mean=%+.4f (q95=%+.4f), null r mean=%+.4f (q95=%.4f)",
                summary_refit["rf"]["empirical_p"], summary_refit["rf"]["null_r2_mean"],
                summary_refit["rf"]["null_r2_q95"], summary_refit["rf"]["null_r_mean"],
                summary_refit["rf"]["null_r_q95"])
        
    # Save outputs
    output_payload = {
        "fixed_chance_reference": summary_fixed,
        "refit_permutation": summary_refit,
    }
    with open(OUT_DIR / "permutation_test_summary.json", "w") as fp:
        json.dump(output_payload, fp, indent=2)
        
    df_rows = []
    for k, v in summary_fixed.items():
        v_copy = dict(v)
        v_copy["type"] = "chance_reference_1000"
        df_rows.append(v_copy)
    for k, v in summary_refit.items():
        v_copy = dict(v)
        v_copy["type"] = "full_refit_loso"
        df_rows.append(v_copy)
    pd.DataFrame(df_rows).to_csv(OUT_DIR / "permutation_results.csv", index=False)
    
    elapsed = time.time() - t_start
    logger.info("Task 5 completed in %.2f seconds.", elapsed)

if __name__ == "__main__":
    run_permutation_tests(n_perm=1000, seed=42)
