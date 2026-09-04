"""Head-to-Head Visualizations Suite.

Generates publication-quality 300 DPI figures comparing MI-EEG vs RS-EEG trial-level task prediction:
- Figure 1: Per-Subject Trial Accuracy Head-to-Head Scatter Plot (MI-EEG vs RS-EEG).
- Figure 2: Trial Prediction Histogram showing subjects achieving >= 75/80 correct trials.
- Figure 3: Head-to-Head Win/Loss/Tie Comparison Bar Chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_head_to_head_scatter(
    df: pd.DataFrame,
    output_path: str = "reports/head_to_head_task_comparison.png"
):
    """Plots scatter plot of MI-EEG trial accuracy vs RS-EEG trial accuracy across subjects."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    x = df['accuracy_lda'].values * 100.0 # MI-EEG CSP+LDA %
    y = df['accuracy_rs_rf'].values * 100.0 # RS-EEG %

    ax.scatter(x, y, color='#0D47A1', alpha=0.7, edgecolors='black', s=50, label='Subjects (N=109)')

    # Identity Line (y = x)
    ax.plot([25, 100], [25, 100], color='#D32F2F', linestyle='--', lw=2, label='Equal Accuracy (y = x)')

    # Target Benchmark Line (75 / 80 trials = 93.75%)
    ax.axhline(93.75, color='#388E3C', linestyle=':', lw=2, label='Target Benchmark (75+/80 = 93.75%)')

    ax.set_xlabel('MI-EEG Direct Task Decoding Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('RS-EEG Guided Task Decoding Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Head-to-Head Task Prediction Accuracy: MI-EEG vs. RS-EEG (109 Subjects)', fontsize=12, fontweight='bold', pad=12)

    ax.set_xlim(25, 105)
    ax.set_ylim(25, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved head-to-head scatter plot to {output_path}")


def plot_trial_correct_distribution(
    df: pd.DataFrame,
    output_path: str = "reports/trial_accuracy_distribution.png"
):
    """Plots distribution of correctly predicted trial counts per subject."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    n_correct_mi = df['n_correct_lda'].values
    n_correct_rs = df['n_correct_rs_rf'].values

    bins = np.linspace(0, df['n_trials'].max(), 20)

    ax.hist(n_correct_mi, bins=bins, color='#1B5E20', alpha=0.6, edgecolor='black', label='MI-EEG Model (CSP+LDA)')
    ax.hist(n_correct_rs, bins=bins, color='#0D47A1', alpha=0.6, edgecolor='black', label='RS-EEG Model (Proposed)')

    # Vertical line at 75 correct trials
    ax.axvline(75, color='#D32F2F', linestyle='--', lw=2.5, label='Benchmark Target (75+ Trials Correct)')

    ax.set_xlabel('Number of Correctly Predicted Motor Task Trials (out of ~80–90 trials)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Trial Task Prediction Performance Across 109 Subjects', fontsize=12, fontweight='bold', pad=12)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trial correct distribution plot to {output_path}")
