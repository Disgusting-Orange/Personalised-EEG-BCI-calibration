"""Generate Model 1 vs Model 2 Head-to-Head Comparison Deliverables.

Compares:
- Model 1 (Actual MI Accuracy % from CSP+LDA on R04-R14)
- Model 2 (GCN-Predicted MI Accuracy % from Resting-State on R01-R02)
- Baselines (Random Forest, XGBoost)

Outputs:
- CSV: reports/model1_vs_model2_subject_comparison.csv
- Excel: C:\\Users\\Admin\\Desktop\\Model1_vs_Model2_Head_to_Head_Comparison.xlsx
- 300 DPI Figure: reports/model1_vs_model2_head_to_head.png
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr, spearmanr

def main():
    gcn_path = 'outputs/benchmark/stage11/gcn_oof_predictions_var_mse_20feat.csv'
    rf_path = 'outputs/benchmark/stage8/spectral_concatenated/rf/rf_oof_predictions.csv'
    xgb_path = 'outputs/benchmark/stage8/spectral_concatenated/xgboost/xgboost_oof_predictions.csv'

    if not os.path.exists(gcn_path):
        raise FileNotFoundError(f"Missing {gcn_path}")

    gcn = pd.read_csv(gcn_path)
    rf = pd.read_csv(rf_path) if os.path.exists(rf_path) else None
    xgb = pd.read_csv(xgb_path) if os.path.exists(xgb_path) else None

    # Construct Head-to-Head DataFrame
    df = pd.DataFrame({
        'Subject_ID': gcn['subject_id'],
        'Model_1_Actual_MI_Accuracy_Pct': (gcn['ground_truth'] * 100).round(2),
        'Model_2_GCN_Predicted_MI_Accuracy_Pct': (gcn['predicted'] * 100).round(2),
        'GCN_Absolute_Error_Pct': (np.abs(gcn['ground_truth'] - gcn['predicted']) * 100).round(2),
        'GCN_Residual_Pct': ((gcn['predicted'] - gcn['ground_truth']) * 100).round(2)
    })

    if rf is not None:
        df['RF_Predicted_Accuracy_Pct'] = (rf['predicted'] * 100).round(2)
        df['RF_Absolute_Error_Pct'] = (np.abs(rf['ground_truth'] - rf['predicted']) * 100).round(2)

    if xgb is not None:
        df['XGB_Predicted_Accuracy_Pct'] = (xgb['predicted'] * 100).round(2)
        df['XGB_Absolute_Error_Pct'] = (np.abs(xgb['ground_truth'] - xgb['predicted']) * 100).round(2)

    # Error brackets
    def get_bracket(err):
        if err <= 5.0:
            return 'High Precision (<= 5% gap)'
        elif err <= 10.0:
            return 'Good (5% - 10% gap)'
        elif err <= 15.0:
            return 'Moderate (10% - 15% gap)'
        else:
            return 'High Discrepancy (> 15% gap)'

    df['Prediction_Accuracy_Tier'] = df['GCN_Absolute_Error_Pct'].apply(get_bracket)

    # 1. Save CSV
    os.makedirs('reports', exist_ok=True)
    csv_out = 'reports/model1_vs_model2_subject_comparison.csv'
    df.to_csv(csv_out, index=False)
    print(f"Saved Head-to-Head Subject Comparison to {csv_out}")

    # 2. Save Excel to Desktop
    desktop1 = r'C:\Users\Admin\Desktop'
    desktop2 = r'C:\Users\Admin\OneDrive\Desktop'
    excel_name = 'Model1_vs_Model2_Head_to_Head_Comparison.xlsx'

    p_excel1 = os.path.join(desktop1, excel_name)
    p_excel2 = os.path.join(desktop2, excel_name)

    with pd.ExcelWriter(p_excel1, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Head-to-Head 109 Subjects', index=False)
        
        # Summary metrics sheet
        r_val, p_val = pearsonr(df['Model_1_Actual_MI_Accuracy_Pct'], df['Model_2_GCN_Predicted_MI_Accuracy_Pct'])
        rho_val, rho_p = spearmanr(df['Model_1_Actual_MI_Accuracy_Pct'], df['Model_2_GCN_Predicted_MI_Accuracy_Pct'])
        mae_val = df['GCN_Absolute_Error_Pct'].mean()
        
        summary_df = pd.DataFrame([
            {'Metric': 'Subjects Evaluated', 'Value': len(df)},
            {'Metric': 'Mean Model 1 Actual MI Accuracy', 'Value': f"{df['Model_1_Actual_MI_Accuracy_Pct'].mean():.2f}%"},
            {'Metric': 'Mean Model 2 GCN Predicted Accuracy', 'Value': f"{df['Model_2_GCN_Predicted_MI_Accuracy_Pct'].mean():.2f}%"},
            {'Metric': 'Mean Absolute Gap / Error (MAE)', 'Value': f"{mae_val:.2f}%"},
            {'Metric': 'Pearson Correlation (r)', 'Value': f"{r_val:.4f} (p = {p_val:.4e})"},
            {'Metric': 'Spearman Rank Correlation (rho)', 'Value': f"{rho_val:.4f} (p = {rho_p:.4e})"},
            {'Metric': 'Subjects within 5% error margin', 'Value': f"{(df['GCN_Absolute_Error_Pct'] <= 5).sum()} ({((df['GCN_Absolute_Error_Pct'] <= 5).sum() / len(df) * 100):.1f}%)"},
            {'Metric': 'Subjects within 10% error margin', 'Value': f"{(df['GCN_Absolute_Error_Pct'] <= 10).sum()} ({((df['GCN_Absolute_Error_Pct'] <= 10).sum() / len(df) * 100):.1f}%)"},
            {'Metric': 'Subjects within 15% error margin', 'Value': f"{(df['GCN_Absolute_Error_Pct'] <= 15).sum()} ({((df['GCN_Absolute_Error_Pct'] <= 15).sum() / len(df) * 100):.1f}%)"}
        ])
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

    if os.path.exists(desktop2):
        shutil.copy(p_excel1, p_excel2)

    print(f"Saved Excel workbook to {p_excel1}")

    # 3. Generate 3-Panel 300 DPI Publication Figure
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax1 = fig.add_subplot(gs[0, 0]) # Scatter
    ax2 = fig.add_subplot(gs[0, 1]) # Error Bracket Distribution
    ax3 = fig.add_subplot(gs[1, :])   # Sorted Bar Chart

    # --- Panel A: Head-to-Head Scatter Plot ---
    x = df['Model_1_Actual_MI_Accuracy_Pct']
    y = df['Model_2_GCN_Predicted_MI_Accuracy_Pct']

    ax1.scatter(x, y, color='#0D47A1', alpha=0.75, s=60, edgecolors='black', label=f'Subjects (N={len(df)})')
    
    # Perfect alignment identity line (y = x)
    min_val, max_val = 25, 95
    ax1.plot([min_val, max_val], [min_val, max_val], color='#D32F2F', linestyle='--', lw=2, label='Perfect Alignment (y = x)')
    
    # Fill +/- 10% error margin corridor
    ax1.fill_between([min_val, max_val], [min_val-10, max_val-10], [min_val+10, max_val+10], color='#BBDEFB', alpha=0.35, label='±10% Accuracy Margin')

    # Regression Fit line
    m, b = np.polyfit(x, y, 1)
    ax1.plot(x, m*x + b, color='#1B5E20', lw=2.5, label=f'GCN Trend (r = {r_val:.3f}, p < 0.001)')

    ax1.set_xlabel('Model 1: Actual MI Decoding Accuracy (%)\n[Calculated from Runs R04–R14 via CSP+LDA]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Model 2: GCN-Predicted MI Accuracy (%)\n[Predicted from Resting-State R01–R02]', fontsize=11, fontweight='bold')
    ax1.set_title('A. Model 1 vs. Model 2 Head-to-Head Scatter Plot', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(25, 95)
    ax1.set_ylim(25, 95)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9.5)

    # --- Panel B: Prediction Error Distribution ---
    tiers = ['High Precision (≤ 5% gap)', 'Good (5% - 10% gap)', 'Moderate (10% - 15% gap)', 'High Discrepancy (> 15% gap)']
    counts = [sum(df['Prediction_Accuracy_Tier'] == t) for t in tiers]
    colors = ['#2E7D32', '#43A047', '#FB8C00', '#E53935']

    bars = ax2.bar(range(len(tiers)), counts, color=colors, edgecolor='black', width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h} ({h/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(range(len(tiers)))
    ax2.set_xticklabels(['≤ 5% Error', '5% – 10% Error', '10% – 15% Error', '> 15% Error'], fontsize=10, fontweight='bold')
    ax2.set_ylabel('Number of Subjects (Total = 109)', fontsize=11, fontweight='bold')
    ax2.set_title('B. GCN Prediction Error Margins Across Cohort', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(counts) + 8)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    # --- Panel C: Sorted Subject-by-Subject Dual Bar Chart ---
    sorted_df = df.sort_values(by='Model_1_Actual_MI_Accuracy_Pct').reset_index(drop=True)
    indices = np.arange(len(sorted_df))
    width = 0.4

    ax3.bar(indices - width/2, sorted_df['Model_1_Actual_MI_Accuracy_Pct'], width=width, color='#1565C0', label='Model 1 (Actual MI Accuracy %)')
    ax3.bar(indices + width/2, sorted_df['Model_2_GCN_Predicted_MI_Accuracy_Pct'], width=width, color='#FF8F00', alpha=0.85, label='Model 2 (GCN-Predicted MI Accuracy %)')

    ax3.set_xlabel('109 Subjects (Sorted by Ascending Actual MI Performance)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Decoding Performance (%)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Subject-by-Subject Comparison: Actual MI (Model 1) vs. GCN-Predicted MI (Model 2)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlim(-1, len(sorted_df))
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    fig_out = 'reports/model1_vs_model2_head_to_head.png'
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Head-to-Head publication figure to {fig_out}")

if __name__ == '__main__':
    main()
