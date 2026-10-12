"""
Fills the red \\fillin{} placeholders in access_revised.tex from the frozen OOF ledger.

Ledger: LOSO_OOF_predictions.csv
Outputs written to: outputs/revision_2026_10/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Map actual ledger column names to model display names
MODELS = {
    'pred_gcn_rf_hybrid': 'Hybrid Ensemble (GCN + RF)',
    'pred_gcn_ensemble': 'GCN Ensemble',
    'pred_base_gcn_alpha': 'Primary GCN',
    'pred_resgcn_dual_band': 'Refined ResGCN',
    'pred_rf_spectral': 'Random Forest',
    'pred_svr_spectral': 'SVR (RBF)',
    'pred_ridge_spectral': 'Ridge Regression',
}

PAIRS = [
    ('pred_gcn_rf_hybrid', 'pred_base_gcn_alpha'),
    ('pred_gcn_rf_hybrid', 'pred_rf_spectral'),
    ('pred_gcn_rf_hybrid', 'pred_ridge_spectral'),
    ('pred_gcn_rf_hybrid', 'pred_svr_spectral'),
    ('pred_base_gcn_alpha', 'pred_rf_spectral'),
    ('pred_base_gcn_alpha', 'mean'),
]

B, SEED = 1000, 0

def metrics(y, p):
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan
    err = y - p
    r2 = 1 - np.sum(err**2) / np.sum((y - y.mean())**2)
    return dict(r=r, R2=r2, MAE=100*np.mean(np.abs(err)), RMSE=100*np.sqrt(np.mean(err**2)))

def mean_baseline(y):               # fold-specific training mean under LOSO
    return (y.sum() - y) / (len(y) - 1)

def ci(a):
    return np.percentile(a, [2.5, 97.5])

def main(path_ledger, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    d = pd.read_csv(path_ledger)
    y = d['actual_mi_accuracy'].to_numpy(float)
    P = {k: d[k].to_numpy(float) for k in MODELS}
    P['mean'] = mean_baseline(y)
    labels = dict(MODELS, mean='Mean Baseline')

    # Verification of Table 4
    print("=" * 80)
    print("VERIFYING TABLE 4 EXACT POINT ESTIMATES:")
    print("=" * 80)
    t4_records = []
    for k in list(MODELS.keys()) + ['mean']:
        m = metrics(y, P[k])
        t4_records.append({
            'Model_Key': k,
            'Model_Name': labels[k],
            'Pearson_r': m['r'],
            'R2': m['R2'],
            'MAE_pp': m['MAE'],
            'RMSE_pp': m['RMSE']
        })
        print(f"{labels[k]:30s} | r={m['r']:.4f} | R2={m['R2']:+.4f} | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f}")
    
    df_t4 = pd.DataFrame(t4_records)
    df_t4.to_csv(out_path / "table4_verified_point_estimates.csv", index=False)

    # Bootstrap CIs (shared resamples)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(B, len(y)))          # shared resamples -> paired bootstrap
    boot = {k: [metrics(y[i], P[k][i]) for i in idx] for k in P}

    print("\n" + "=" * 80)
    print(f"TABLE CI ROWS (B={B}, SEED={SEED}) (point [lo, hi]):")
    print("=" * 80)
    ci_records = []
    latex_ci_rows = []
    for k in list(MODELS.keys()) + ['mean']:
        pt = metrics(y, P[k])
        rec = {'Model_Key': k, 'Model_Name': labels[k]}
        row = []
        for m in ['r', 'R2', 'MAE', 'RMSE']:
            if k == 'mean' and m == 'r':
                row.append('--')
                rec[f'{m}_point'] = np.nan
                rec[f'{m}_ci_lo'] = np.nan
                rec[f'{m}_ci_hi'] = np.nan
                continue
            lo, hi = ci([b[m] for b in boot[k]])
            rec[f'{m}_point'] = pt[m]
            rec[f'{m}_ci_lo'] = lo
            rec[f'{m}_ci_hi'] = hi
            if m in ('R2',):
                cell = f'{pt[m]:+.4f} [{lo:+.4f}, {hi:+.4f}]'
            elif m == 'r':
                cell = f'{pt[m]:.4f} [{lo:.4f}, {hi:.4f}]'
            else:
                cell = f'{pt[m]:.2f} [{lo:.2f}, {hi:.2f}]'
            row.append(cell)
        line = labels[k] + ' & ' + ' & '.join(row) + r' \\'
        print(line)
        latex_ci_rows.append(line)
        ci_records.append(rec)
    
    df_ci = pd.DataFrame(ci_records)
    df_ci.to_csv(out_path / "table_ci_bootstrap_1000.csv", index=False)
    with open(out_path / "table_ci_rows.tex", "w") as fp:
        fp.write("\n".join(latex_ci_rows) + "\n")

    # Paired differences
    print("\n" + "=" * 80)
    print("TABLE PAIRED DIFFERENCES (first minus second; 95% CI):")
    print("=" * 80)
    paired_records = []
    latex_paired_rows = []
    for a, b_ in PAIRS:
        cells = []
        pair_name = f"{labels[a]} - {labels[b_]}"
        rec = {'Comparison': pair_name, 'Model_A': labels[a], 'Model_B': labels[b_]}
        for m, f in [('R2', '{:+.4f}'), ('MAE', '{:+.2f}'), ('RMSE', '{:+.2f}')]:
            diff = np.array([x[m] - z[m] for x, z in zip(boot[a], boot[b_])])
            pt = metrics(y, P[a])[m] - metrics(y, P[b_])[m]
            lo, hi = ci(diff)
            rec[f'Delta_{m}_point'] = pt
            rec[f'Delta_{m}_ci_lo'] = lo
            rec[f'Delta_{m}_ci_hi'] = hi
            cells.append(f'{f.format(pt)} [{f.format(lo)}, {f.format(hi)}]')
        line = f'{pair_name} &' + ' & '.join(cells) + r' \\'
        print(line)
        latex_paired_rows.append(line)
        paired_records.append(rec)
        
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(out_path / "table_paired_differences.csv", index=False)
    with open(out_path / "table_paired_rows.tex", "w") as fp:
        fp.write("\n".join(latex_paired_rows) + "\n")

    # Quartiles
    print("\n" + "=" * 80)
    print("QUARTILES BY OBSERVED TARGET (MAE / Bias in pp):")
    print("=" * 80)
    q = np.percentile(y, [25, 50, 75])
    bins = np.digitize(y, q)
    quartile_records = []
    latex_quartile_rows = []
    q_names = ['Q1 (Low)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (High)']
    q_ranges = ['$34.3$--$53.7\\%$', '$53.7$--$61.3\\%$', '$61.3$--$72.7\\%$', '$72.7$--$98.9\\%$']
    for g, name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        m = bins == g
        true_mean = 100 * np.mean(y[m])
        rec = {
            'Quartile': name,
            'Count': int(m.sum()),
            'True_Mean_Pct': true_mean,
            'Observed_Min_Pct': 100 * np.min(y[m]),
            'Observed_Max_Pct': 100 * np.max(y[m])
        }
        out = []
        for k in ['pred_base_gcn_alpha', 'pred_gcn_rf_hybrid', 'mean']:
            k_mae = 100 * np.mean(np.abs(P[k][m] - y[m]))
            k_bias = 100 * np.mean(P[k][m] - y[m])
            k_short = 'gcn' if 'gcn_alpha' in k else ('hybrid' if 'hybrid' in k else 'mean')
            rec[f'{k_short}_mae'] = k_mae
            rec[f'{k_short}_bias'] = k_bias
            out += [f'{k_mae:.2f}', f'{k_bias:+.1f}']
        
        line = f"{q_names[g]} & {q_ranges[g]} & {true_mean:.1f}\\% & " + " & ".join(out) + r" \\"
        print(f"{name} (n={m.sum()}) true_mean={true_mean:.1f}% & " + " & ".join(out) + r" \\")
        latex_quartile_rows.append(line)
        quartile_records.append(rec)
        
    df_quartiles = pd.DataFrame(quartile_records)
    df_quartiles.to_csv(out_path / "table_quartile_analysis.csv", index=False)
    with open(out_path / "table_quartile_rows.tex", "w") as fp:
        fp.write("\n".join(latex_quartile_rows) + "\n")

    # Error bands
    print("\n" + "=" * 80)
    print("ERROR BANDS (Share of subjects, n):")
    print("=" * 80)
    error_band_records = []
    edges = [(0, 5), (5, 10), (10, 15), (15, np.inf)]
    for k in ['pred_gcn_rf_hybrid', 'mean']:
        e = 100 * np.abs(P[k] - y)
        k_name = labels[k]
        band_strs = []
        for lo, hi in edges:
            cond = (e > lo) & (e <= hi) if lo > 0 else (e <= hi)
            pct = 100 * np.mean(cond)
            cnt = int(cond.sum())
            band_strs.append(f"{lo}-{hi}pp: {pct:.1f}% (n={cnt})")
            error_band_records.append({
                'Model': k_name,
                'Band_Low': lo,
                'Band_High': hi,
                'Percent': pct,
                'Count': cnt
            })
        med_e = np.median(e)
        print(f"{k_name}: " + "; ".join(band_strs))
        print(f"  median abs err: {med_e:.2f} pp")
        
    df_error_bands = pd.DataFrame(error_band_records)
    df_error_bands.to_csv(out_path / "table_error_bands.csv", index=False)

if __name__ == '__main__':
    ledger = sys.argv[1] if len(sys.argv) > 1 else 'LOSO_OOF_predictions.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'outputs/revision_2026_10'
    main(ledger, out)
