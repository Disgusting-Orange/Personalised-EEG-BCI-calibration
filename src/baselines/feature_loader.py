"""Feature loading and preparation module for Stage 8 Classical Baseline Regressors.

Loads frozen outputs from Stage 3 (MI continuous targets), Stage 6 (Spectral node features),
and Stage 7 (wPLI connectivity node degrees), constructing leak-free dataset arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("baselines.feature_loader")

# Channel region mappings for regional aggregation
REGION_MAP = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "AF3", "AF4", "AF7", "AF8", "F1", "F2", "F5", "F6"],
    "Central": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
    "Parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P5", "P3", "P1", "Pz", "P2", "P4", "P6"],
    "Occipital": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "CB1", "CB2"],
    "Temporal": ["FT7", "FT8", "T7", "T8", "TP7", "TP8"],
}


def load_dataset(
    feature_set: str = "spectral_concatenated",
    subjects: Sequence[str] | None = None,
    targets_dir: Path | str | None = None,
    features_dir: Path | str | None = None,
    connectivity_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load continuous targets and feature vectors for specified subjects.

    Parameters
    ----------
    feature_set:
        Feature representation ('spectral_concatenated', 'spectral_plus_wpli', 'regional_aggregated').
    subjects:
        Optional sequence of subject IDs (e.g. ['S001', 'S002']). Defaults to all available subjects.
    targets_dir, features_dir, connectivity_dir:
        Optional directory paths to Stage 3, Stage 6, and Stage 7 outputs.

    Returns
    -------
    X: np.ndarray
        Feature matrix of shape (n_subjects, n_features).
    y: np.ndarray
        Target vector of continuous balanced accuracy scores of shape (n_subjects,).
    subject_ids: list[str]
        List of subject ID strings corresponding to matrix rows.
    feature_names: list[str]
        List of feature column names.
    """
    targets_root = Path(targets_dir) if targets_dir else REPOSITORY_ROOT / "outputs" / "targets"
    features_root = Path(features_dir) if features_dir else REPOSITORY_ROOT / "outputs" / "features"
    conn_root = Path(connectivity_dir) if connectivity_dir else REPOSITORY_ROOT / "outputs" / "connectivity"

    if subjects is None:
        target_sub_dirs = sorted(list(targets_root.glob("stage3_s*")))
        subjects = [d.name.replace("stage3_", "").upper() for d in target_sub_dirs]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    valid_subjects: list[str] = []
    feature_names: list[str] = []

    for sub_id in subjects:
        sub_lower = sub_id.lower()
        target_file = targets_root / f"stage3_{sub_lower}" / "mi_targets.csv"
        eo_pow_file = features_root / f"stage6_{sub_lower}" / f"{sub_id}_R01_band_powers.csv"
        ec_pow_file = features_root / f"stage6_{sub_lower}" / f"{sub_id}_R02_band_powers.csv"

        if not target_file.exists() or not eo_pow_file.exists() or not ec_pow_file.exists():
            logger.warning("Skipping subject %s: missing target or spectral feature file.", sub_id)
            continue

        # Load Target
        df_target = pd.read_csv(target_file)
        y_val = float(df_target["balanced_accuracy"].iloc[0])

        # Load Spectral Features
        df_eo = pd.read_csv(eo_pow_file, index_col=0)
        df_ec = pd.read_csv(ec_pow_file, index_col=0)

        # Spectral feature vectors
        eo_features = df_eo.values.flatten()
        ec_features = df_ec.values.flatten()
        spec_vector = np.concatenate([eo_features, ec_features])

        if not feature_names:
            bands = list(df_eo.columns)
            channels = list(df_eo.index)
            eo_names = [f"EO_{ch}_{b}" for ch in channels for b in bands]
            ec_names = [f"EC_{ch}_{b}" for ch in channels for b in bands]
            spec_names = eo_names + ec_names
        else:
            bands = list(df_eo.columns)
            channels = list(df_eo.index)
            spec_names = feature_names[: len(spec_vector)]

        if feature_set == "spectral_concatenated":
            row_features = spec_vector
            if not feature_names:
                feature_names = spec_names

        elif feature_set == "spectral_plus_wpli":
            wpli_eo_file = conn_root / f"stage7_{sub_lower}" / f"{sub_id}_R01_alpha_wpli.npy"
            wpli_ec_file = conn_root / f"stage7_{sub_lower}" / f"{sub_id}_R02_alpha_wpli.npy"

            if not wpli_eo_file.exists() or not wpli_ec_file.exists():
                logger.warning("Skipping subject %s for wPLI feature set: missing connectivity file.", sub_id)
                continue

            wpli_eo = np.load(wpli_eo_file)
            wpli_ec = np.load(wpli_ec_file)

            # Node degree = sum of off-diagonal wPLI weights
            deg_eo = np.sum(wpli_eo, axis=1)
            deg_ec = np.sum(wpli_ec, axis=1)
            conn_vector = np.concatenate([deg_eo, deg_ec])

            row_features = np.concatenate([spec_vector, conn_vector])
            if not feature_names:
                eo_conn_names = [f"EO_wPLI_alpha_{ch}" for ch in channels]
                ec_conn_names = [f"EC_wPLI_alpha_{ch}" for ch in channels]
                feature_names = spec_names + eo_conn_names + ec_conn_names

        elif feature_set == "regional_aggregated":
            reg_vector_list: list[float] = []
            reg_names_list: list[str] = []

            for cond_prefix, df_pow in [("EO", df_eo), ("EC", df_ec)]:
                for reg_name, reg_channels in REGION_MAP.items():
                    valid_chs = [ch for ch in reg_channels if ch in df_pow.index]
                    if valid_chs:
                        reg_mean_pow = df_pow.loc[valid_chs].mean(axis=0)
                    else:
                        reg_mean_pow = pd.Series(0.0, index=df_pow.columns)

                    for b in df_pow.columns:
                        reg_vector_list.append(float(reg_mean_pow[b]))
                        if not feature_names:
                            reg_names_list.append(f"{cond_prefix}_{reg_name}_{b}")

            row_features = np.array(reg_vector_list, dtype=np.float64)
            if not feature_names:
                feature_names = reg_names_list

        else:
            raise ValueError(f"Unknown feature set '{feature_set}'. Supported: 'spectral_concatenated', 'spectral_plus_wpli', 'regional_aggregated'")

        X_list.append(row_features)
        y_list.append(y_val)
        valid_subjects.append(sub_id)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    logger.info("Loaded dataset '%s': X.shape=%s y.shape=%s for %d subjects", feature_set, X.shape, y.shape, len(valid_subjects))
    return X, y, valid_subjects, feature_names
