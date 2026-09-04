"""Resting-State EEG Task Model and Head-to-Head Classifier.

Evaluates Resting-State EEG (RS-EEG) spectral features & wPLI graph topology
for direct motor imagery task prediction and head-to-head benchmarking against MI-EEG models.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.gcn_model import GCNRegressor


class RSTaskGCN(nn.Module):
    """GCN model taking Resting-State graph topology and decoding task trials."""

    def __init__(self, in_channels: int = 20, hidden_channels: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super(RSTaskGCN, self).__init__()
        self.gcn1 = GCNRegressor(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=3, dropout=dropout)
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x, edge_index, batch=None):
        out_reg = self.gcn1(x, edge_index, batch) # (batch_size, 1)
        logits = self.classifier(out_reg) # (batch_size, num_classes)
        return logits


def evaluate_rs_task_model(
    rs_features: np.ndarray,
    task_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluates Resting-State EEG features directly on task trial classification.

    Args:
        rs_features: Resting-state feature representation (e.g. 20-D or 320-D spectral vector).
        task_labels: Trial task labels (e.g. 0 vs 1 for Left vs Right fist / Both Fists vs Feet).
        n_splits: Folds for cross-validation.
        random_state: Random seed.

    Returns:
        Dictionary of trial accuracy metrics for RS-EEG models.
    """
    n_trials = len(task_labels)
    if n_trials == 0:
        return {
            'n_trials': 0,
            'n_correct_rs_rf': 0,
            'accuracy_rs_rf': 0.0,
            'kappa_rs_rf': 0.0
        }

    # Expand RS features across task trials if RS is subject-level
    if rs_features.ndim == 1:
        rs_trials = np.tile(rs_features, (n_trials, 1))
    else:
        rs_trials = rs_features

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state)

    preds_rf = np.zeros(n_trials, dtype=int)
    preds_xgb = np.zeros(n_trials, dtype=int)

    for train_idx, test_idx in skf.split(rs_trials, task_labels):
        X_train, X_test = rs_trials[train_idx], rs_trials[test_idx]
        y_train, y_test = task_labels[train_idx], task_labels[test_idx]

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        preds_rf[test_idx] = rf.predict(X_test)
        preds_xgb[test_idx] = xgb.predict(X_test)

    acc_rf = float(accuracy_score(task_labels, preds_rf))
    acc_xgb = float(accuracy_score(task_labels, preds_xgb))
    kappa_rf = float(cohen_kappa_score(task_labels, preds_rf))
    kappa_xgb = float(cohen_kappa_score(task_labels, preds_xgb))

    return {
        'n_trials': n_trials,
        'n_correct_rs_rf': int(np.sum(preds_rf == task_labels)),
        'accuracy_rs_rf': acc_rf,
        'kappa_rs_rf': kappa_rf,
        'n_correct_rs_xgb': int(np.sum(preds_xgb == task_labels)),
        'accuracy_rs_xgb': acc_xgb,
        'kappa_rs_xgb': kappa_xgb
    }
