#!/usr/bin/env python3
"""Compose the 5774 multi-teacher MOPD config shape.

Usage:
    python scripts/mopd_cfg_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.config import omega_conf_to_dataclass


def compose_from_overrides(config_dir: Path, override_string: str):
    overrides = override_string.split()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="ppo_trainer.yaml", overrides=overrides)


def main() -> int:
    repo = REPO_ROOT
    config_dir = repo / "verl" / "trainer" / "config"

    if not config_dir.is_dir():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        return 2

    multi_teacher_overrides = (
        "distillation.enabled=False "
        "+distillation.teacher_models.geo3k.task=geo3k "
        "+distillation.teacher_models.geo3k.model_path=path/to/geo3k_teacher "
        "+distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1 "
        "+distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.3 "
        "+distillation.teacher_models.gsm8k.task=gsm8k "
        "+distillation.teacher_models.gsm8k.model_path=path/to/gsm8k_teacher "
        "+distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1 "
        "+distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.3 "
    )

    multi_teacher_cfg = compose_from_overrides(config_dir, multi_teacher_overrides)
    pprint(omega_conf_to_dataclass(multi_teacher_cfg.distillation))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
