"""
config.py
---------
Central configuration for the EEG preprocessing pipeline.
All tunable parameters live here — edit this file to change behaviour
without touching any other module.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_DIR.parent / "preprocessed"
LOG_DIR    = DATA_DIR.parent / "logs"
FIGURE_DIR = DATA_DIR.parent / "figures"

# ── PhysioNet run-type mapping ────────────────────────────────────────────────
TASK_MAP = {
    "R01": "baseline_eyes_open",
    "R02": "baseline_eyes_closed",
    "R03": "execution_left_right_fist",
    "R04": "imagery_left_right_fist",
    "R05": "execution_both_fists_feet",
    "R06": "imagery_both_fists_feet",
    "R07": "execution_left_right_fist",
    "R08": "imagery_left_right_fist",
    "R09": "execution_both_fists_feet",
    "R10": "imagery_both_fists_feet",
    "R11": "execution_left_right_fist",
    "R12": "imagery_left_right_fist",
    "R13": "execution_both_fists_feet",
    "R14": "imagery_both_fists_feet",
}

# ── Filtering ────────────────────────────────────────────────────────────────
BANDPASS_LOW   = 1.0
BANDPASS_HIGH  = 40.0

# PhysioNet EEGMMI was recorded in the USA
NOTCH_FREQ     = 60.0
NOTCH_WIDTH    = 2.0

# ── ICA ──────────────────────────────────────────────────────────────────────
ICA_N_COMPONENTS      = 20
ICA_METHOD            = "infomax"
ICA_MAX_ITER          = 1000
ICA_RANDOM_STATE      = 42

# Frontal channels used as EOG proxies
ICA_EOG_CHANNELS      = ["Fp1", "Fp2"]

# Correlation threshold for blink detection
ICA_EOG_THRESHOLD     = 3.0

# Slightly relaxed for PhysioNet EEG
ICA_MUSCLE_THRESHOLD  = 2.0

# High-pass used only for ICA fitting
ICA_HIGH_PASS_FOR_FIT = 1.0

# ── Bad-channel detection ────────────────────────────────────────────────────
BAD_CHANNEL_STD_MULT = 2.0

# Flat if std < 0.5 µV
BAD_CHANNEL_FLAT_STD = 0.5e-6

# ── Epoching ─────────────────────────────────────────────────────────────────
EPOCH_DURATION = 2.0
EPOCH_TMIN     = 0.0

# --------------------------------------------------------------------------
# IMPORTANT:
#
# Roadmap requirement:
#     Reject epochs exceeding ±100 µV
#
# MNE interprets reject={'eeg': x}
# as PEAK-TO-PEAK amplitude.
#
# ±100 µV corresponds to:
#
# (+100 µV) - (-100 µV)
#      = 200 µV peak-to-peak
#
# Therefore:
#     200e-6 V
# --------------------------------------------------------------------------

EPOCH_REJECT = {
    "eeg": 250e-6
}

# Flat-line threshold
EPOCH_FLAT = {
    "eeg": 1e-6
}

# ── Re-referencing ──────────────────────────────────────────────────────────
REFERENCE = "average"

# ── Output ──────────────────────────────────────────────────────────────────
SAVE_FORMAT = "fif"

NUMPY_DTYPE = "float32"

OVERWRITE = False

# ── Visualisation ───────────────────────────────────────────────────────────
VIZ_CHANNELS = [
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "O1",
    "O2",
]

VIZ_DURATION = 10.0
VIZ_N_EPOCHS = 4

SAVE_FIGURES = False
SHOW_FIGURES = False

# ── Processing ──────────────────────────────────────────────────────────────
N_JOBS = -1

RANDOM_SEED = 42

TARGET_SFREQ = 128