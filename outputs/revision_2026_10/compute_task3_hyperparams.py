"""
Extract and compute:
1. Retained 2-s epochs per resting recording (218 recordings) after +-100 uV rejection.
2. Selected Ridge alpha and SVR C/epsilon across the 109 outer folds (median and range).
3. wPLI specification from code.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

def compute_epoch_stats(qc_dir: Path):
    usable_epochs = []
    generated_epochs = []
    rejected_epochs = []
    records = []

    for sub_id in range(1, 110):
        for r in ['R01', 'R02']:
            f = qc_dir / f'S{sub_id:03d}_{r}_qc_report.json'
            with open(f) as fp:
                data = json.load(fp)
            m = data['metrics']
            usable = m['n_usable_epochs']
            gen = m['n_generated_epochs']
            rej = m['n_rejected_epochs']
            usable_epochs.append(usable)
            generated_epochs.append(gen)
            rejected_epochs.append(rej)
            records.append({'subject': f'S{sub_id:03d}', 'run': r, 'generated': gen, 'usable': usable, 'rejected': rej})

    usable = np.array(usable_epochs)
    df_epochs = pd.DataFrame(records)
    
    stats_dict = {
        'total_recordings': len(usable),
        'mean_retained': float(np.mean(usable)),
        'min_retained': int(np.min(usable)),
        'max_retained': int(np.max(usable)),
        'std_retained': float(np.std(usable)),
        'total_rejected': int(np.sum(rejected_epochs))
    }
    return stats_dict, df_epochs

def tune_ridge_and_svr(X, y, loso_splits_path: Path):
    with open(loso_splits_path) as fp:
        splits = json.load(fp)
    
    ridge_grid = {'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    svr_grid = {
        'regressor__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'regressor__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'regressor__epsilon': [0.001, 0.01, 0.1]
    }
    
    ridge_best_alphas = []
    svr_best_Cs = []
    svr_best_epsilons = []
    svr_best_gammas = []
    
    n_folds = len(splits)
    print(f"Running nested 5-fold CV for Ridge across {n_folds} outer folds...")
    for i, split in enumerate(splits):
        train_idx = split['train_idx']
        X_tr, y_tr = X[train_idx], y[train_idx]
        
        # Ridge
        pipe_r = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(random_state=42 + i))])
        cv_r = KFold(n_splits=5, shuffle=True, random_state=42 + i)
        search_r = GridSearchCV(pipe_r, ridge_grid, cv=cv_r, scoring='neg_mean_squared_error', n_jobs=-1)
        search_r.fit(X_tr, y_tr)
        ridge_best_alphas.append(search_r.best_params_['regressor__alpha'])
        
        # SVR
        pipe_s = Pipeline([('scaler', StandardScaler()), ('regressor', SVR())])
        cv_s = KFold(n_splits=5, shuffle=True, random_state=42 + i)
        search_s = GridSearchCV(pipe_s, svr_grid, cv=cv_s, scoring='neg_mean_squared_error', n_jobs=-1)
        search_s.fit(X_tr, y_tr)
        svr_best_Cs.append(search_s.best_params_['regressor__C'])
        svr_best_epsilons.append(search_s.best_params_['regressor__epsilon'])
        svr_best_gammas.append(search_s.best_params_['regressor__gamma'])
        
        if (i + 1) % 25 == 0 or (i + 1) == n_folds:
            print(f"  Completed outer fold {i+1}/{n_folds}")
            
    df_hparams = pd.DataFrame({
        'fold': list(range(1, n_folds + 1)),
        'ridge_alpha': ridge_best_alphas,
        'svr_C': svr_best_Cs,
        'svr_epsilon': svr_best_epsilons,
        'svr_gamma': svr_best_gammas
    })
    
    ridge_stats = {
        'median_alpha': float(np.median(ridge_best_alphas)),
        'min_alpha': float(np.min(ridge_best_alphas)),
        'max_alpha': float(np.max(ridge_best_alphas)),
        'value_counts': pd.Series(ridge_best_alphas).value_counts().to_dict()
    }
    
    svr_stats = {
        'median_C': float(np.median(svr_best_Cs)),
        'min_C': float(np.min(svr_best_Cs)),
        'max_C': float(np.max(svr_best_Cs)),
        'C_counts': pd.Series(svr_best_Cs).value_counts().to_dict(),
        'median_epsilon': float(np.median(svr_best_epsilons)),
        'min_epsilon': float(np.min(svr_best_epsilons)),
        'max_epsilon': float(np.max(svr_best_epsilons)),
        'epsilon_counts': pd.Series(svr_best_epsilons).value_counts().to_dict(),
        'gamma_counts': pd.Series(svr_best_gammas).value_counts().to_dict(),
    }
    
    return ridge_stats, svr_stats, df_hparams

def main():
    out_dir = Path("outputs/revision_2026_10")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Epoch stats
    print("Computing epoch statistics across 218 recordings...")
    qc_dir = Path("outputs/preprocessed/qc")
    epoch_stats, df_epochs = compute_epoch_stats(qc_dir)
    print("Epoch Stats:", epoch_stats)
    df_epochs.to_csv(out_dir / "resting_epochs_218_recordings.csv", index=False)
    with open(out_dir / "epoch_stats_summary.json", "w") as fp:
        json.dump(epoch_stats, fp, indent=2)
        
    # 2. Hyperparameter selection across 109 outer folds
    feat_dir = Path('outputs/features')
    X_list = []
    rel_cols = ['delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel']
    for i in range(1, 110):
        sub = f'S{i:03d}'
        sub_l = sub.lower()
        f_eo = feat_dir / f'stage6_{sub_l}' / f'{sub}_R01_band_powers.csv'
        f_ec = feat_dir / f'stage6_{sub_l}' / f'{sub}_R02_band_powers.csv'
        eo_vals = pd.read_csv(f_eo, index_col=0)[rel_cols].values.flatten()
        ec_vals = pd.read_csv(f_ec, index_col=0)[rel_cols].values.flatten()
        X_list.append(np.concatenate([eo_vals, ec_vals]))
    X = np.array(X_list)
    
    d = pd.read_csv('LOSO_OOF_predictions.csv')
    y = d['actual_mi_accuracy'].values
    
    loso_splits_path = Path("outputs/benchmark/loso_splits.json")
    ridge_stats, svr_stats, df_hparams = tune_ridge_and_svr(X, y, loso_splits_path)
    print("\nRidge Stats:", ridge_stats)
    print("SVR Stats:", svr_stats)
    
    df_hparams.to_csv(out_dir / "selected_hyperparameters_109_folds.csv", index=False)
    with open(out_dir / "selected_hyperparameters_summary.json", "w") as fp:
        json.dump({'ridge': ridge_stats, 'svr': svr_stats}, fp, indent=2)

if __name__ == '__main__':
    main()
