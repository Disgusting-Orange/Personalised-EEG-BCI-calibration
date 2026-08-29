"""Technical execution of the FROZEN Stage 2 preprocessing pipeline for S001/R01.

This regenerates the R01 resting-state artifacts at the current pipeline
specification (128 Hz resampled), replacing the earlier R01 artifacts that
were preprocessed at 160 Hz under a prior config version.

This is NOT a Stage 2 implementation change. Stage 2 source code is frozen
and is NOT modified by this script. This script only INVOKES the existing
frozen pipeline functions with the current preprocessing.yaml config.

Per AGENTS.md Section 4a, this is a technical-execution event, distinct from
the frozen state of Stage 2 itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from preprocessing.artifacts import (  # noqa: E402
    detect_artifact_components,
    detect_bad_channels,
    run_bad_channel_pipeline,
    run_ica_pipeline,
)
from preprocessing.epochs import run_epoch_pipeline  # noqa: E402
from preprocessing.filters import run_filter_pipeline  # noqa: E402
from preprocessing.qc import run_qc_pipeline  # noqa: E402

SUBJECT_ID = "S001"
RUN_ID = "R01"


def _resolve_input_edf(config: dict, repository_root: Path) -> Path:
    dataset_root = repository_root / config["dataset_root"]
    return dataset_root / SUBJECT_ID / f"{SUBJECT_ID}{RUN_ID}.edf"


def main() -> None:
    repository_root = REPOSITORY_ROOT
    config_path = repository_root / "configs" / "preprocessing.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    input_edf = _resolve_input_edf(config, repository_root)
    if not input_edf.exists():
        raise FileNotFoundError(f"Input EDF not found: {input_edf}")

    preprocessed_root = repository_root / "outputs" / "preprocessed"

    print(f"[Stage 2 technical execution] subject={SUBJECT_ID} run={RUN_ID}")
    print(f"  input_edf={input_edf}")

    # 1. Filtering (notch + bandpass + resample), per frozen preprocessing.yaml.
    filtered_raw = run_filter_pipeline(
        input_edf,
        subject=SUBJECT_ID,
        run=RUN_ID,
        config=config,
        save_filtered=True,
        output_dir=preprocessed_root / "filtered",
        overwrite=True,
    )

    # 2. Bad-channel detection + interpolation.
    bad_channels = detect_bad_channels(filtered_raw, config=config)
    bad_channel_raw = run_bad_channel_pipeline(
        filtered_raw,
        subject=SUBJECT_ID,
        run=RUN_ID,
        config=config,
        save_interpolated=False,
        output_dir=preprocessed_root,
        overwrite=True,
    )

    # 3. ICA fitting + artifact removal.
    ica_raw, ica_object = run_ica_pipeline(
        bad_channel_raw,
        subject=SUBJECT_ID,
        run=RUN_ID,
        config=config,
        save_ica_model=True,
        output_dir=preprocessed_root / "ica",
        overwrite=True,
    )
    ica_components_removed: list[int] = []
    if ica_object is not None:
        ica_components_removed = detect_artifact_components(
            ica_object, bad_channel_raw, config=config
        )

    # 4. Fixed-length epoch extraction.
    epochs, _epochs_path = run_epoch_pipeline(
        ica_raw,
        subject=SUBJECT_ID,
        run=RUN_ID,
        config=config,
        save_epochs_output=True,
        output_dir=preprocessed_root / "epochs",
        overwrite=True,
    )

    # 5. QC reporting.
    run_qc_pipeline(
        raw=ica_raw,
        epochs=epochs,
        subject=SUBJECT_ID,
        run=RUN_ID,
        bad_channels=bad_channels,
        ica_components_removed=ica_components_removed,
        config=config,
        output_dir=preprocessed_root / "qc",
    )

    print(f"[Stage 2 technical execution] complete for subject={SUBJECT_ID} run={RUN_ID}")


if __name__ == "__main__":
    main()
