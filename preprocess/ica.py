# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 RUNNER: Local PhysioNet → ICA → 8–30 Hz → Epochs → Save
# ─────────────────────────────────────────────────────────────────────────────

import sys

# -----------------------------
# CONFIG
# -----------------------------

SUBJECTS = list(range(16, 110))   # starts from Subject 16 to 109

DATA_ROOT = Path(
    "/home/nvidia/22BLC1376/data/eegmmidb/"
    "eeg-motor-movementimagery-dataset-1.0.0/files"
)

BASE_SAVE_DIR = Path("saved_step1_all_subjects")
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [3, 4, 7, 8, 11, 12]

# Epoch window
TMIN = 0.5
TMAX = 3.5

# Filtering
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 40.0

MOTOR_BAND_LOW = 8.0
MOTOR_BAND_HIGH = 30.0

NOTCH_FREQ = 60.0
REFERENCE = "average"

# ICA
ICA_N_COMPONENTS = 20
ICA_METHOD = "infomax"
ICA_MAX_ITER = 1000
ICA_RANDOM_STATE = 42
ICA_EOG_CHANNELS = ["Fp1", "Fp2"]
ICA_EOG_THRESHOLD = 3.0
ICA_MUSCLE_THRESHOLD = 2.0
ICA_HIGH_PASS_FOR_FIT = 1.0

N_JOBS = 1

# Labels
LABEL_MAP = {
    "rest": 0,
    "executed_left": 1,
    "executed_right": 2,
    "imagined_left": 3,
    "imagined_right": 4,
}

LABEL_NAMES = {
    0: "Rest",
    1: "Executed Left",
    2: "Executed Right",
    3: "Imagined Left",
    4: "Imagined Right",
}


# -----------------------------
# LOGGER
# -----------------------------

def make_logger():
    logger = logging.getLogger("ICA_STEP1")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = make_logger()


# -----------------------------
# LOCAL FILE PATHS
# -----------------------------

def get_local_file_paths(subject_id: int):
    subject_folder = DATA_ROOT / f"S{subject_id:03d}"

    file_paths = [
        subject_folder / f"S{subject_id:03d}R{run:02d}.edf"
        for run in RUNS
    ]

    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing EDF file: {path}")

    return file_paths


# -----------------------------
# LOAD ONE SUBJECT
# -----------------------------

def load_subject_step1(subject_id: int):
    file_paths = get_local_file_paths(subject_id)

    all_epochs = []
    all_labels = []

    for run, file_path in zip(RUNS, file_paths):
        print(f"\nProcessing Subject {subject_id}, Run {run}...")
        print(f"Using file: {file_path}")

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Standardize channel names
        mne.datasets.eegbci.standardize(raw)

        # Montage
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

        # Average reference
        raw.set_eeg_reference(REFERENCE, verbose=False)

        # Notch filter
        raw.notch_filter(
            freqs=NOTCH_FREQ,
            verbose=False,
        )

        # Broad filter before ICA: 1–40 Hz
        raw.filter(
            l_freq=BANDPASS_LOW,
            h_freq=BANDPASS_HIGH,
            fir_design="firwin",
            verbose=False,
        )

        # ICA cleaning using your pipeline
        raw_clean, fitted_ica, excluded_components = run_ica_pipeline(
            raw_filtered=raw,
            n_components=ICA_N_COMPONENTS,
            method=ICA_METHOD,
            max_iter=ICA_MAX_ITER,
            random_state=ICA_RANDOM_STATE,
            eog_channels=ICA_EOG_CHANNELS,
            eog_threshold=ICA_EOG_THRESHOLD,
            muscle_threshold=ICA_MUSCLE_THRESHOLD,
            hp_cutoff=ICA_HIGH_PASS_FOR_FIT,
            n_jobs=N_JOBS,
            logger=logger,
            figure_dir=None,
            subject_id=f"S{subject_id:03d}",
            run_id=f"R{run:02d}",
        )

        print(f"Run {run}: ICA removed components {excluded_components}")

        # Final motor-band filter: 8–30 Hz
        raw_clean.filter(
            l_freq=MOTOR_BAND_LOW,
            h_freq=MOTOR_BAND_HIGH,
            fir_design="firwin",
            verbose=False,
        )

        # Events
        events, event_id = mne.events_from_annotations(raw_clean, verbose=False)

        used_event_id = {
            key: value
            for key, value in event_id.items()
            if key in ["T0", "T1", "T2"]
        }

        if len(used_event_id) == 0:
            print(f"Run {run}: No T0/T1/T2 events found. Skipping.")
            continue

        # Epoching
        epochs = mne.Epochs(
            raw_clean,
            events,
            event_id=used_event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X_run = epochs.get_data()
        event_codes = epochs.events[:, -1]

        inv_event_id = {v: k for k, v in used_event_id.items()}

        y_run = []

        for code in event_codes:
            event_name = inv_event_id[code]

            if event_name == "T0":
                label = LABEL_MAP["rest"]

            elif run in [3, 7, 11]:
                if event_name == "T1":
                    label = LABEL_MAP["executed_left"]
                elif event_name == "T2":
                    label = LABEL_MAP["executed_right"]
                else:
                    continue

            elif run in [4, 8, 12]:
                if event_name == "T1":
                    label = LABEL_MAP["imagined_left"]
                elif event_name == "T2":
                    label = LABEL_MAP["imagined_right"]
                else:
                    continue

            y_run.append(label)

        if len(y_run) != len(X_run):
            print(f"Warning: Run {run} X/y mismatch: {len(X_run)} vs {len(y_run)}")

        all_epochs.append(X_run)
        all_labels.extend(y_run)

        print(f"Run {run} done. Epochs: {len(y_run)}")

    if len(all_epochs) == 0:
        raise ValueError(f"No valid epochs found for Subject {subject_id}")

    X = np.concatenate(all_epochs, axis=0)
    y = np.array(all_labels)

    return X, y


# -----------------------------
# RUN ALL SUBJECTS FROM 16
# -----------------------------

if __name__ == "__main__":

    failed_subjects = []

    for subject_id in SUBJECTS:
        print("\n" + "=" * 60)
        print(f"Starting Subject {subject_id}")
        print("=" * 60)

        subject_save_dir = BASE_SAVE_DIR / f"subject_{subject_id:03d}"
        subject_save_dir.mkdir(parents=True, exist_ok=True)

        x_path = subject_save_dir / "X_epochs_ica_cleaned_motor_8_30.npy"
        y_path = subject_save_dir / "y_labels.npy"

        if x_path.exists() and y_path.exists():
            print(f"Subject {subject_id} already processed. Skipping.")
            continue

        try:
            X, y = load_subject_step1(subject_id)

            print("X shape:", X.shape)
            print("y shape:", y.shape)

            unique, counts = np.unique(y, return_counts=True)

            print("Class counts:")
            for cls, count in zip(unique, counts):
                print(f"{LABEL_NAMES[cls]}: {count}")

            np.save(x_path, X)
            np.save(y_path, y)

            print(f"Saved Subject {subject_id} to {subject_save_dir}")

        except Exception as e:
            print(f"Subject {subject_id} failed: {e}")
            failed_subjects.append(subject_id)

    failed_path = BASE_SAVE_DIR / "failed_subjects_from_16.txt"

    with open(failed_path, "w") as f:
        for sub in failed_subjects:
            f.write(str(sub) + "\n")

    print("\nAll done.")
    print("Failed subjects:", failed_subjects)
    print("Failed list saved to:", failed_path)