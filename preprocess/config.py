"""
config.py
---------
Central configuration for the EEG preprocessing pipeline.
All tunable parameters live here — edit this file to change behaviour
without touching any other module.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_DIR  : folder that contains S001/, S002/, … subject folders
# OUTPUT_DIR: where preprocessed epochs are written
DATA_DIR   = Path(__file__).resolve().parents[1]          # …/files/
OUTPUT_DIR = DATA_DIR.parent / "preprocessed"
LOG_DIR    = DATA_DIR.parent / "logs"
FIGURE_DIR = DATA_DIR.parent / "figures"

# ── PhysioNet run-type mapping ─────────────────────────────────────────────────
# R01, R02     → baseline (eyes open / eyes closed)
# R03,R07,R11  → motor imagery: left / right fist
# R04,R08,R12  → motor execution: left / right fist
# R05,R09,R13  → motor imagery: both fists / both feet
# R06,R10,R14  → motor execution: both fists / both feet
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

# ── Filtering ──────────────────────────────────────────────────────────────────
BANDPASS_LOW   = 0.5    # Hz
BANDPASS_HIGH  = 45.0   # Hz
NOTCH_FREQ     = 60.0   # Hz  (US powerline; change to 50 for EU/Asian datasets)
NOTCH_WIDTH    = 2.0    # Hz  bandwidth around notch freq

# ── ICA ────────────────────────────────────────────────────────────────────────
ICA_N_COMPONENTS     = 20       # number of ICA components to decompose
ICA_METHOD           = "fastica"
ICA_MAX_ITER         = 800
ICA_RANDOM_STATE     = 42
# EOG-proxy channels in this dataset (frontal channels used as EOG proxies)
ICA_EOG_CHANNELS     = ["Fp1", "Fp2"]
ICA_EOG_THRESHOLD    = 3.0      # z-score threshold for EOG correlation
ICA_MUSCLE_THRESHOLD = 1.0      # z-score threshold for muscle artefacts
ICA_HIGH_PASS_FOR_FIT = 1.0     # Hz — high-pass applied before ICA fitting

# ── Bad-channel detection ──────────────────────────────────────────────────────
BAD_CHANNEL_STD_MULT  = 5.0    # channels > N × median(std) across channels
BAD_CHANNEL_FLAT_STD  = 0.5e-6 # channels with std < threshold (flat line)

# ── Epoching ──────────────────────────────────────────────────────────────────
EPOCH_DURATION = 2.0    # seconds (non-overlapping)
EPOCH_TMIN     = 0.0    # epoch start relative to window start
# Amplitude rejection threshold (peak-to-peak per epoch)
EPOCH_REJECT   = {"eeg": 550e-6}   # 450 µV
EPOCH_FLAT     = {"eeg": 1e-6}     # 1 µV  (flat-line detection)

# ── Re-referencing ─────────────────────────────────────────────────────────────
REFERENCE = "average"   # common average reference

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_FORMAT  = "fif"       # "fif" saves MNE Epochs objects
NUMPY_DTYPE  = "float32"   # used when SAVE_FORMAT == "numpy"
OVERWRITE    = False       # skip already-processed files when False

# ── Visualisation ──────────────────────────────────────────────────────────────
VIZ_CHANNELS  = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
VIZ_DURATION  = 10.0   # seconds of signal to show in comparison plot
VIZ_N_EPOCHS  = 4      # number of epochs shown in epoch comparison plot
SAVE_FIGURES  = True
SHOW_FIGURES  = False  # set True for interactive / Jupyter use

# ── Processing ─────────────────────────────────────────────────────────────────
N_JOBS      = -1    # parallel jobs for MNE operations (-1 = all cores)
RANDOM_SEED = 42
