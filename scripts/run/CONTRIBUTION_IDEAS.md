# Contribution Ideas for verl

This document outlines potential contribution opportunities for the verl project (v0.7.0). Ideas are organized by difficulty level and area of focus.

## Quick Links

- [Good First Issues](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Call for Contribution](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)
- [RFC Issues](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3ARFC)

---

## Deprecation Notice

**Important**: The following files are part of the legacy engine and should NOT be worked on for new contributions. Focus efforts on the new engine instead.

| File | Engine | Trainer | Backend | Notes |
|------|--------|---------|---------|-------|
| `fsdp_sft_trainer.py` | Legacy | SFT | FSDP | Deprecated |
| `fsdp_workers.py` | Legacy | RL | FSDP | Deprecated |
| `megatron_workers.py` | Legacy | RL | Megatron | Deprecated |
| `sft_trainer.py` | **New** | SFT | fsdp/megatron/veomni/... | torchrun SPMD |
| `sft_trainer_ray.py` | **New** | SFT | fsdp/megatron/veomni/... | Ray single controller |
| `engine_workers.py` | **New** | RL | fsdp/megatron/veomni/... | Active development |

**Focus contributions on**: `engine_workers.py`, `sft_trainer.py`, `sft_trainer_ray.py`, and `verl/workers/engine/`

---

## Beginner-Friendly Contributions

### 1. Remove Deprecated Checkpoint Loaders
**Difficulty**: Easy | **Impact**: Low

Clean up deprecated files:
- `verl/models/qwen2/megatron/checkpoint/qwen2_loader_depracated.py`
- `verl/models/llama/megatron/checkpoint/llama_loader_depracated.py`

Verify the new loaders work and remove the deprecated versions.

---

## Medium-Difficulty Contributions

### 2. Improve New Engine Worker Features
**Difficulty**: Medium | **Impact**: High

Enhance the new `engine_workers.py` with additional features like hybrid sharding and memory optimization.

**Location**: `verl/workers/engine_workers.py`

**Features to add**:
- FSDP hybrid shard support for larger models
- Memory profiling extensions
- Better config alignment between actor/ref models

**Reference**: `verl/workers/engine/` for engine implementations.

---

### 3. Expand Optimizer Configuration in New Engine
**Difficulty**: Medium | **Impact**: Medium

Add more optimizer arguments to the config system for the new engine.

**Location**: `verl/workers/config/`

**Reference**: `verl/workers/engine_workers.py` for how config is consumed.

---

### 4. Enable SGLang Router
**Difficulty**: Medium | **Impact**: Medium

Enable the SGLang router for the HTTP server engine.

**Location**: `verl/workers/rollout/sglang_rollout/http_server_engine.py`
```python
# TODO: @ChangyiYang Enable SGLang router for this http server engine
```

---

### 5. Video Input Support for SGLang
**Difficulty**: Medium | **Impact**: High

Add video input support for SGLang rollout.

**Location**: `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
```python
# TODO: support video input for sglang
```

**Context**: v0.7 added video input optimization; SGLang needs to catch up.

---

### 6. Unify LoRA Config Across Backends
**Difficulty**: Medium | **Impact**: Medium

Unify FSDP and Megatron LoRA configurations.

**Location**: `verl/workers/config/model.py`
```python
# TODO: unify fsdp and megatron lora config
```

---

## Advanced Contributions

### 7. VLM with MoE Support
**Difficulty**: Hard | **Impact**: High

Add support for Vision-Language Models with Mixture of Experts.

**Location**: `verl/workers/actor/megatron_actor.py`
```python
# TODO: support VLM with MoE
```

---

### 8. Complete FSDP2 Support in New Engine
**Difficulty**: Hard | **Impact**: High

Ensure full FSDP2 support in the new engine for better performance.

**Location**: `verl/workers/engine/fsdp/`

**Context**: Part of v0.8 roadmap (switch default to new model engine). The new engine should fully leverage FSDP2 capabilities.

**Reference**: `verl/workers/engine_workers.py` for the new engine entry point.

---

### 9. Reference Model Parameter Offload in New Engine
**Difficulty**: Hard | **Impact**: Medium-High

Implement/improve reference model parameter offloading in the new engine.

**Location**: `verl/workers/engine_workers.py`

**Context**: Memory optimization for training larger models with reference models.

---

### 10. Implement TensorRT-LLM Rollout Engine
**Difficulty**: Hard | **Impact**: High

Add TensorRT-LLM as a new rollout engine.

**Context**: Listed in v0.8 roadmap as new rollout engine.

**Reference**:
- `verl/workers/rollout/vllm_rollout/` - vLLM implementation
- `verl/workers/rollout/sglang_rollout/` - SGLang implementation

---

### 11. Stabilize VLA (Vision-Language-Action) Module
**Difficulty**: Hard | **Impact**: High

Help stabilize the experimental VLA module for robotics/embodied AI.

**Location**: `verl/experimental/vla/`

**Features to develop**:
- Isaac environment integration
- LIBERO dataset support
- End-to-end training pipeline

---

### 12. Fully Async Policy Training
**Difficulty**: Hard | **Impact**: High

Help finalize the fully async policy training for merging to main.

**Location**: `verl/experimental/fully_async_policy/`

**Context**: v0.9 roadmap plans to "Merge Full async into main".

---

## Roadmap Alignment

These contributions align with the official roadmap:

### v0.8 Targets
- Deprecate DataProto by Tensordict
- Switch default to new model engine (`engine_workers.py`)
- Feature parity between new and legacy engines (#2, #3, #8, #9)
- Polish VeOmni engine (`verl/workers/engine/veomni/`)
- TensorRT-LLM rollout engine (#10)

### v0.9 Targets
- Merge Full async into main (#12)
- Remove legacy model engine (`fsdp_workers.py`, `megatron_workers.py`)
- Support omni-model RL training
- Agentic training recipes (SWEAgent, GUIAgent)

---

## Getting Started

1. Fork the repository
2. Set up development environment: `pip install -e .[test,vllm]`
3. Install pre-commit hooks: `pre-commit install`
4. Pick an issue from this list or GitHub
5. Create a branch and submit a PR

See [CONTRIBUTING.md](/CONTRIBUTING.md) for detailed guidelines.

---

## Notes

- Start with beginner-friendly issues to understand the codebase
- Join discussions on RFC issues for major features
- Check recent commits for code style and patterns
- Run tests locally before submitting PRs
