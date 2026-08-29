"""Run the Stage 5 resting-state preprocessing validation workflow.

Validates frozen Stage 2 resting-state outputs (R01 eyes-open, R02 eyes-closed)
for S001. Produces JSON + Markdown reports with the AGENTS.md Section 8
methodological evaluation of EO vs EC recordings.

Stage 5 is validation/QC only — no re-preprocessing, no features, no ML.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from resting_state import run_stage5  # noqa: E402


if __name__ == "__main__":
    config_path = REPOSITORY_ROOT / "configs" / "stage5_resting_state_validation.yaml"
    summary = run_stage5(config_path)

    for subject in summary["subjects"]:
        print(
            f"{subject['subject_id']}: verdict={subject['verdict']} "
            f"runs_validated={len(subject['runs'])}"
        )
