#!/usr/bin/env python3
"""Minimal Hydra compose repro for indexed distillation teacher overrides.

Usage:
    python scripts/repro_hydra_teacher_model_index_override.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> int:
    repo = Path.cwd().resolve()
    config_dir = repo / "verl" / "trainer" / "config"
    example_dir = repo / "examples" / "on_policy_distillation_trainer"

    if not config_dir.is_dir():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        return 2
    if not example_dir.is_dir():
        print(f"ERROR: Example directory not found: {example_dir}", file=sys.stderr)
        return 2

    overrides = [
        f"hydra.searchpath=[file://{example_dir}]",
        "+config@distillation.teacher_model=teacher_model",
        "distillation.teacher_model[0].model_path=foo",
    ]

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            compose(config_name="ppo_trainer.yaml", overrides=overrides)
    except Exception as exc:
        print("PASS: Hydra rejected the indexed override at compose time.")
        print(f"{type(exc).__name__}: {exc}")
        return 0

    print("FAIL: Hydra accepted the indexed override unexpectedly.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
