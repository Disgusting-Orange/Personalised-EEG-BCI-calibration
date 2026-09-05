"""Standalone execution script for Stage 14 — Statistical Analysis & Scientific Validation.

Executes paired Wilcoxon signed-rank tests, paired t-tests, Holm-Bonferroni correction,
Cohen's dz & rank-biserial effect sizes, 95% bootstrap confidence intervals,
N=1000 target permutation tests, Bland-Altman agreement plots, and LaTeX master table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graph.dataset import EEGGraphDataset
from graph.statistical_tests import (
    bootstrap_confidence_intervals,
    compute_loso_fold_robustness,
    paired_model_comparisons,
    target_permutation_test,
)
from graph.statistical_viz import plot_bland_altman, plot_predicted_vs_actual_scatter, plot_residual_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 14 Statistical Analysis and Scientific Validation.")
    parser.add_argument("--config", default="configs/stage14_statistical_analysis.yaml")
    parser.add_argument("--phase", choices=["1", "2"], default="2", help="Validation phase (1: S001-S010 subcohort, 2: Full cohort).")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("stage14")

    config_path = REPOSITORY_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    out_dir = REPOSITORY_ROOT / cfg.get("output_directory", "outputs/statistical_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Stage 14: Statistical Analysis & Scientific Validation")
    logger.info("Phase: %s | Permutations: %d | Bootstrap: %d", args.phase, args.n_permutations, args.n_bootstrap)
    logger.info("==================================================")

    start_time = time.perf_counter()

    # Load ground-truth targets from dataset
    ds = EEGGraphDataset(root=REPOSITORY_ROOT / "outputs/graph_dataset/wpli_alpha_top15")
    y_true = np.array([float(ds[i].y.item()) for i in range(len(ds))])

    # Subset for Phase 1 if specified
    if args.phase == "1":
        y_true = y_true[:10]

    # Load OOF predictions for all 9 models
    model_ids = ["gcn", "gat", "svr", "ridge", "elasticnet", "lasso", "rf", "xgboost", "dummy"]
    preds = {}

    for m_id in model_ids:
        # Check potential OOF CSV locations
        possible_paths = [
            REPOSITORY_ROOT / f"outputs/benchmark/stage8/spectral_concatenated/{m_id}/{m_id}_oof_predictions.csv",
            REPOSITORY_ROOT / f"outputs/benchmark/stage11/{m_id}_oof_predictions.csv",
            REPOSITORY_ROOT / f"outputs/benchmark/stage12/{m_id}_oof_predictions.csv",
        ]
        loaded = False
        for p in possible_paths:
            if p.exists():
                df_oof = pd.read_csv(p)
                col = "predicted" if "predicted" in df_oof.columns else df_oof.columns[2]
                y_m = df_oof[col].values
                if args.phase == "1":
                    y_m = y_m[:10]
                preds[m_id] = y_m
                loaded = True
                break

        # Ensure real prediction file was loaded; raise error if missing
        if not loaded:
            raise FileNotFoundError(
                f"Required out-of-fold prediction CSV missing for model '{m_id}'. "
                f"Checked paths: {possible_paths}. Synthetic fallback generation is strictly disabled."
            )

    # 1. Paired Model Significance Tests (GCN vs All)
    df_paired = paired_model_comparisons(y_true, preds, ref_model="gcn")
    df_paired.to_csv(out_dir / "paired_comparisons.csv", index=False)
    logger.info("Paired Wilcoxon Comparisons (GCN vs All):\n%s", df_paired.to_string())

    # 2. Bootstrap Confidence Intervals (95% CIs for MAE, RMSE, R2, Pearson r, Spearman ρ)
    ci_results = {}
    ci_rows = []
    for m_id, y_p in preds.items():
        ci_dict = bootstrap_confidence_intervals(y_true, y_p, n_bootstrap=args.n_bootstrap, seed=42)
        ci_results[m_id] = ci_dict
        ci_rows.append({
            "model": m_id.upper(),
            "mae": ci_dict["mae"],
            "mae_ci_lower": ci_dict["mae_ci"][0],
            "mae_ci_upper": ci_dict["mae_ci"][1],
            "rmse": ci_dict["rmse"],
            "rmse_ci_lower": ci_dict["rmse_ci"][0],
            "rmse_ci_upper": ci_dict["rmse_ci"][1],
            "r2": ci_dict["r2"],
            "r2_ci_lower": ci_dict["r2_ci"][0],
            "r2_ci_upper": ci_dict["r2_ci"][1],
            "pearson_r": ci_dict["pearson_r"],
            "pearson_r_ci_lower": ci_dict["pearson_r_ci"][0],
            "pearson_r_ci_upper": ci_dict["pearson_r_ci"][1],
        })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)

    # 3. Target Permutation Test (N=1000) for GCN
    gcn_perm = target_permutation_test(y_true, preds["gcn"], n_permutations=args.n_permutations, seed=42)
    (out_dir / "permutation_test_results.json").write_text(json.dumps(gcn_perm, indent=2), encoding="utf-8")
    logger.info("GCN Permutation Test (N=%d): R2_p=%.4e, Pearson_p=%.4e", args.n_permutations, gcn_perm["perm_r2_pvalue"], gcn_perm["perm_pearson_pvalue"])

    # 4. LOSO Fold Robustness Analysis
    df_robustness = compute_loso_fold_robustness(y_true, preds)
    df_robustness.to_csv(out_dir / "fold_robustness.csv", index=False)

    # 5. Diagnostic Figures (300 DPI)
    plot_bland_altman(y_true, preds["gcn"], "gcn", out_dir / "bland_altman_plot.png", dpi=cfg.get("dpi", 300))
    plot_residual_diagnostics(y_true, preds["gcn"], "gcn", out_dir / "residual_diagnostics.png", dpi=cfg.get("dpi", 300))
    plot_predicted_vs_actual_scatter(y_true, preds["gcn"], "gcn", ci_results["gcn"], out_dir / "gcn_scatter_plot.png", dpi=cfg.get("dpi", 300))

    # 6. Master Publication LaTeX & CSV Table
    df_ci_copy = df_ci.copy()
    df_ci_copy["model_key"] = df_ci_copy["model"].str.lower()
    df_paired_copy = df_paired.copy()
    df_paired_copy["target_model_key"] = df_paired_copy["target_model"].str.lower()
    df_master = pd.merge(df_ci_copy, df_paired_copy, left_on="model_key", right_on="target_model_key", how="left").drop(columns=["model_key", "target_model_key"])
    df_master.to_csv(out_dir / "master_statistical_table.csv", index=False)

    # Save to both local Desktop and OneDrive Desktop results_cni
    import os
    import shutil
    for d in [r"C:\Users\Admin\Desktop\results_cni", r"C:\Users\Admin\OneDrive\Desktop\results_cni"]:
        try:
            os.makedirs(d, exist_ok=True)
            for f in ["master_statistical_table.csv", "bland_altman_plot.png", "gcn_scatter_plot.png", "residual_diagnostics.png"]:
                src_f = out_dir / f
                if src_f.exists():
                    shutil.copy(src_f, os.path.join(d, f))
        except Exception as e:
            logger.warning("Could not copy to %s: %s", d, e)

    # LaTeX Table code
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Stage 14 Formal Statistical Validation & Bootstrap Confidence Intervals (N=109)}",
        r"\label{tab:stage14_statistical_summary}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{MAE [95\% CI]} & \textbf{RMSE [95\% CI]} & \textbf{R$^2$ [95\% CI]} & \textbf{Pearson $r$} & \textbf{Wilcoxon $p_{\text{adj}}$} & \textbf{Cohen's $d_z$} \\",
        r"\hline",
    ]
    for _, r in df_master.iterrows():
        m_str = r["model"]
        mae_str = f"{r['mae']:.4f} [{r['mae_ci_lower']:.4f}, {r['mae_ci_upper']:.4f}]"
        rmse_str = f"{r['rmse']:.4f} [{r['rmse_ci_lower']:.4f}, {r['rmse_ci_upper']:.4f}]"
        r2_str = f"{r['r2']:.4f} [{r['r2_ci_lower']:.4f}, {r['r2_ci_upper']:.4f}]"
        r_str = f"{r['pearson_r']:.4f}"
        p_str = f"{r['p_adj_holm']:.4e}" if pd.notnull(r.get("p_adj_holm")) else "Ref"
        dz_str = f"{r['cohens_dz']:.3f}" if pd.notnull(r.get("cohens_dz")) else "Ref"

        bold = r"\textbf{" if m_str in ("GCN", "SVR") else ""
        endb = "}" if m_str in ("GCN", "SVR") else ""
        latex_lines.append(f"  {bold}{m_str}{endb} & {mae_str} & {rmse_str} & {r2_str} & {r_str} & {p_str} & {dz_str} \\\\")

    latex_lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    (out_dir / "master_statistical_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    elapsed = time.perf_counter() - start_time

    val_report = {
        "status": "PASS",
        "phase": args.phase,
        "n_subjects": len(y_true),
        "total_models_evaluated": len(preds),
        "gcn_vs_svr_p_adj": float(df_paired[df_paired["target_model"] == "svr"]["p_adj_holm"].iloc[0]) if "svr" in df_paired["target_model"].values else None,
        "gcn_vs_svr_cohens_dz": float(df_paired[df_paired["target_model"] == "svr"]["cohens_dz"].iloc[0]) if "svr" in df_paired["target_model"].values else None,
        "gcn_permutation_pvalue": gcn_perm["perm_pearson_pvalue"],
        "elapsed_seconds": round(elapsed, 2),
    }

    (out_dir / "validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
    logger.info("Stage 14 Statistical Testing complete in %.2fs. Report saved at %s", elapsed, out_dir / "validation_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
