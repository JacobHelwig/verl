roadmap
v0.7 release
Model Engine

Integrate Megatron-Bridge and support LoRA/PEFT, see blog post: How We Build Trillion Parameter Reasoning RL with 10% GPUs

Support experimental fp8 training for megatron backend

Support new model for megatron backend: GPT-OSS, Qwen3-Next

Comprehensive support for new mode engine, FSDP and Megatron engine are production ready.

Dispatch tensordict with nested tensor instead of padded DataProto

Add TrainingWorker that resembles Tinker-like API

Add VLM support for model engine, SFT and RL trainer

Add model engine based critic model

Implement ActorRolloutRefWorker by TrainingWorker, support different backend in one worker

New VeOmni engine added, still in alpha status.

Rollout Engine

Remove SPMD rollout mode

Support blockwise fp8 rollout for vllm and sglang; support online quant for vllm with torchao

Experimental router replay support for vllm

Optimize multi-modal data fetch and preprocess, support video input

Upgrade to vllm==0.12.0; sglang==0.5.6

Reward

Support hybrid reward scenarios, including generative, discriminative, rule-based rewards, and their combinations.

Refactor reward models into server mode, supporting both colocated and standalone deployments.

Introduce new reward managers to handle more complex scenarios, limited mode for request rate control and remote mode for CPU-intensive tasks.

Algorithm

Add CISPO: Clipped IS-weight Policy Optimization

Add SAPO: Soft Adaptive Policy Optimization

Recipe

[NEW] VLA: add experimental support for VLA model

[NEW] rhymerl: History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL

TransferQueue: support multiple data partition and optimize tensor zero-copy serialization

One-step-off-policy/Fully async: optimize weight synchronization by checkpoint engine with bucket and pipeline support.

v0.8
Model Engine

Deprecate DataProto by Tensordict for zero padding transmission

Switch default to new model engine, mark legacy engine (fsdp_workers.py, megatron_workers.py) as deprecated

Feature parity between new and legacy model engine: LoRA/PEFT, etc

Polish VeOmni engine to production ready status

Support MTP RL training

Optimize GPU memory for long context: fine-grained activation recompuation/offload

New model support: DeepSeek V3.2, etc

Rollout Engine

New rollout engine TensorRT-LLM

Separate vllm worker from trainer process, update weights by cuda ipc

TransferQueue

Merge TransferQueue recipe into main

Optimize e2e image/video vlm training pipeline by TransferQueue

Optimize router replay transmission by TransferQueue

Checkpoint Engine

Add checkpoint engine abstract interface

Add NCCL and NIXL transport backend

Add more transport backend

v0.9
Trainer

Merge Full async into main: refactor with verl-core component

Model Engine

Remove legacy model engine (fsdp_workers.py, megatron_workers.py)

Support omni-model RL training: Qwen3-Omni, BAGEL, etc

Rollout Engine

New rollout engine vllm-omni

More agentic training recipe

SWEAgent

GUIAgent