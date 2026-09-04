"""Unit and Integration Tests for Stage 17 Head-to-Head Task Prediction Benchmark."""

import pytest
import numpy as np
import pandas as pd
from src.mi_decoding.trial_decoder import decode_subject_trials
from src.graph.rs_task_model import evaluate_rs_task_model
from src.graph.head_to_head_viz import plot_head_to_head_scatter, plot_trial_correct_distribution

def test_evaluate_rs_task_model():
    # Synthetic trial dataset
    n_trials = 80
    rs_features = np.random.randn(320)
    task_labels = np.random.choice([0, 1], size=n_trials)

    res = evaluate_rs_task_model(rs_features, task_labels, n_splits=3)

    assert res['n_trials'] == 80
    assert 'n_correct_rs_rf' in res
    assert 'accuracy_rs_rf' in res
    assert 0.0 <= res['accuracy_rs_rf'] <= 1.0

def test_head_to_head_viz(tmp_path):
    df_dummy = pd.DataFrame({
        'subject_id': ['S001', 'S002', 'S003'],
        'n_trials': [80, 80, 80],
        'n_correct_lda': [76, 60, 78],
        'accuracy_lda': [0.95, 0.75, 0.975],
        'n_correct_rs_rf': [50, 45, 52],
        'accuracy_rs_rf': [0.625, 0.5625, 0.65],
        'meets_75_target': [True, False, True]
    })

    p_scatter = str(tmp_path / "test_scatter.png")
    p_dist = str(tmp_path / "test_dist.png")

    plot_head_to_head_scatter(df_dummy, p_scatter)
    plot_trial_correct_distribution(df_dummy, p_dist)

    assert (tmp_path / "test_scatter.png").exists()
    assert (tmp_path / "test_dist.png").exists()
