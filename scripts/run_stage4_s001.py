"""Run the approved Stage 4 target-quality validation workflow.

Stage 4 validates Stage 3 target outputs. It is a thin entry point over
the reusable validator in src/target_quality/validator.py — it runs over
every subject discovered in the configured target directory, so it works
unchanged once multi-subject target generation exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from target_quality import run_stage4  # noqa: E402


if __name__ == "__main__":
    config_path = REPOSITORY_ROOT / "configs" / "stage4_target_quality.yaml"
    summary = run_stage4(config_path)

    print(f"Stage 4 validation complete. Target directory: {summary['target_directory']}")
    print(f"Subjects validated: {len(summary['subjects'])}")
    for subject in summary["subjects"]:
        print(
            f"  {subject['subject_id']}: verdict={subject['verdict']} "
            f"target={subject.get('target_value')}"
        )
