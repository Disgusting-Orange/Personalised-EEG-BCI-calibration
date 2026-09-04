"""Trial-Level Motor Imagery Task Decoder.

Performs trial-level motor imagery task classification across PhysioNet EEGMMIDB runs:
- Task Group 1 (R04, R08, R12): Left Fist vs Right Fist imagery (~45 trials/subject)
- Task Group 2 (R06, R10, R14): Both Fists vs Both Feet imagery (~45 trials/subject)
- Combined Task Group (R04-R14 imagery): All Motor Imagery (~90 trials/subject)

Returns exact trial counts (N_correct / N_trials), Accuracy %, Balanced Accuracy,
Cohen's Kappa, and F1-score per subject.
"""

import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from mne.decoding import CSP


def load_subject_edf(raw_dir: str, subject_id: int, run_num: int) -> Tuple[mne.io.BaseRaw, Any, Any]:
    """Locates and loads raw EDF file for subject and run."""
    sub_str = f"S{subject_id:03d}"
    run_str = f"R{run_num:02d}"
    edf_path = Path(raw_dir) / sub_str / f"{sub_str}{run_str}.edf"

    if not edf_path.exists():
        # Fallback recursive search if nested differently
        matches = list(Path(raw_dir).rglob(f"{sub_str}{run_str}.edf"))
        if matches:
            edf_path = matches[0]
        else:
            raise FileNotFoundError(f"EDF file not found: {sub_str}{run_str}.edf")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    return raw


def decode_subject_trials(
    raw_dir: str,
    subject_id: int,
    runs: List[int],
    tmin: float = 0.5,
    tmax: float = 3.5,
    n_csp_components: int = 4,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Extracts trial epochs and evaluates trial-level motor task prediction.

    Args:
        raw_dir: Path to raw dataset directory.
        subject_id: Subject index (1-109).
        runs: List of run numbers (e.g. [4, 8, 12] or [6, 10, 14]).
        tmin: Epoch start time in seconds relative to event.
        tmax: Epoch end time in seconds relative to event.
        n_csp_components: Number of CSP filters.
        n_splits: Folds for cross-validation.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing trial counts, accuracy metrics, and model performances.
    """
    epochs_list = []
    labels_list = []

    for run_num in runs:
        try:
            raw = load_subject_edf(raw_dir, subject_id, run_num)
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            t1_t2_id = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
            if len(t1_t2_id) < 2:
                continue

            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=t1_t2_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
                verbose=False
            )

            if len(epochs) > 0:
                X_run = epochs.get_data() # (n_trials, n_channels, n_times)
                y_run = epochs.events[:, -1]
                unique_labels = sorted(np.unique(y_run))
                label_map = {l: idx for idx, l in enumerate(unique_labels)}
                y_mapped = np.array([label_map[l] for l in y_run])

                epochs_list.append(X_run)
                labels_list.append(y_mapped)
        except Exception:
            continue

    if not epochs_list:
        return {
            'subject_id': f"S{subject_id:03d}",
            'n_trials': 0,
            'n_correct_lda': 0,
            'accuracy_lda': 0.0,
            'balanced_accuracy_lda': 0.0,
            'kappa_lda': 0.0,
            'f1_lda': 0.0
        }

    X_all = np.concatenate(epochs_list, axis=0) # (total_trials, n_channels, n_times)
    y_all = np.concatenate(labels_list, axis=0)
    n_trials = len(y_all)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        'lda': LinearDiscriminantAnalysis(),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'xgboost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state),
        'svm': SVC(kernel='rbf', C=1.0, random_state=random_state)
    }

    preds = {m: np.zeros(n_trials, dtype=int) for m in models}

    for train_idx, test_idx in skf.split(X_all, y_all):
        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        csp = CSP(n_components=n_csp_components, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train_raw, y_train)
        X_test_csp = csp.transform(X_test_raw)

        for m_name, model in models.items():
            model.fit(X_train_csp, y_train)
            preds[m_name][test_idx] = model.predict(X_test_csp)

    res = {
        'subject_id': f"S{subject_id:03d}",
        'n_trials': n_trials
    }

    for m_name in models:
        y_pred = preds[m_name]
        n_correct = int(np.sum(y_pred == y_all))
        acc = accuracy_score(y_all, y_pred)
        b_acc = balanced_accuracy_score(y_all, y_pred)
        kappa = cohen_kappa_score(y_all, y_pred)
        f1 = f1_score(y_all, y_pred, average='macro')

        res[f'n_correct_{m_name}'] = n_correct
        res[f'accuracy_{m_name}'] = float(acc)
        res[f'balanced_accuracy_{m_name}'] = float(b_acc)
        res[f'kappa_{m_name}'] = float(kappa)
        res[f'f1_{m_name}'] = float(f1)

    return res
