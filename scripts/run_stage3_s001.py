"""Run the approved Stage 3 S001-only MI target-generation workflow."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mi_decoding.target_generation import run_stage3_target_generation


if __name__ == "__main__":
    outputs = run_stage3_target_generation(REPOSITORY_ROOT / "configs" / "stage3_mi_decoder.yaml")
    for name, path in outputs.items():
        print(f"{name}: {path}")
