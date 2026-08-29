"""Technical execution of the FROZEN Stage 2 preprocessing pipeline for S001/R02.

This is NOT a Stage 2 implementation change. Stage 2 source code is frozen
and is NOT modified by this script. This script only INVOKES the existing
frozen pipeline functions for the one resting-state run (R02 / eyes-closed)
that was never queued during the original Stage 2 execution.

Per AGENTS.md Section 4a, this is a technical-execution event, distinct from
the frozen state of Stage 2 itself.

The script:
  * loads the unmodified configs/preprocessing.yaml
  * invokes the frozen run_*_pipeline functions in the same order used to
    produce R01 outputs
  * passes no new parameters; every parameter comes from preprocessing.yaml
  * writes the same artifact set produced for R01:
      - filtered raw
      - ICA model
      - epochs
      - QC report + summary + plot
      - validation log entries appended to outputs/preprocessed/*_validation.log

Bad-channel and ICA-component lists are captured via the frozen public helper
functions (detect_bad_channels, detect_artifact_components) so the QC report
populates the same `n_bad_channels_detected` and `ica_components_removed`
fields present in the R01 QC report. This mirrors how the R01 QC report was
populated; it does not alter Stage 2 behaviour.
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
RUN_ID = "R02"


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
    #    Capture the detected list separately for the QC report, since the
    #    frozen run_bad_channel_pipeline returns only the cleaned Raw.
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
    #    Returns (cleaned_raw, ica_object). Components removed are captured
    #    via the frozen detect_artifact_components helper for the QC report.
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
        ica_components_removed = detect_artifact_components(ica_object, bad_channel_raw, config=config)

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
