#!/usr/bin/env python3
"""Hydra compose for distillation teacher list overrides.

Usage:
    python scripts/mopd_cfg_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def run_compose(config_dir: Path, override_string: str) -> tuple[bool, str]:
    overrides = override_string.split()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            compose(config_name="ppo_trainer.yaml", overrides=overrides)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "compose succeeded"


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

    full_list_overrides = (
        f"hydra.searchpath=[file://{example_dir}] "
        "+config@distillation.teacher_model=teacher_model "
        "distillation.teacher_model=[{_target_:verl.workers.config.DistillationTeacherModelConfig,"
        "task_name:geo3k,model_path:foo,"
        "inference:{_target_:verl.workers.config.RolloutConfig,name:vllm,prompt_length:16,response_length:8,"
        "max_model_len:64,max_num_batched_tokens:64,tensor_model_parallel_size:1,engine_kwargs:{}}},"
        "{_target_:verl.workers.config.DistillationTeacherModelConfig,task_name:gsm8k,model_path:bar,"
        "inference:{_target_:verl.workers.config.RolloutConfig,name:vllm,prompt_length:16,response_length:8,"
        "max_model_len:64,max_num_batched_tokens:64,tensor_model_parallel_size:1,engine_kwargs:{}}}]"
    )
    indexed_overrides = (
        f"hydra.searchpath=[file://{example_dir}] "
        "+config@distillation.teacher_model=teacher_model "
        "distillation.teacher_model[0].model_path=foo"
    )

    for label, override_string in (
        ("full_list", full_list_overrides),
        ("indexed", indexed_overrides),
    ):
        ok, message = run_compose(config_dir, override_string)
        print(f"{label}: {'[SUCCESS]' if ok else '[FAILURE]'}")
        print(f"overrides: {override_string}")
        print(f"result: {message}")
        print("\n" + "=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
