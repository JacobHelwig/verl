#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR=$PWD/data
export HF_HOME=$DATA_DIR
export PATH="$CONDA_PREFIX/bin:$PATH"
export VLLM_CACHE_ROOT=$DATA_DIR/vllm_cache
set -x

# use_dynamic_bsz=False
# max_token_len=8192
#     actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
#     actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
#     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
#     actor_rollout_ref.actor.ppo_max_token_'len_per_gpu=${max_token_len} \
#     actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len} \
#     actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len} \


gsm8k_train_path=$DATA_DIR/gsm8k/train.parquet
gsm8k_test_path=$DATA_DIR/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

ppo_micro_batch_size_per_gpu=2
ppo_micro_batch_size_per_gpu=8

branch=main
branch=simpleFixEngineMetrics

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=0 \
    data.return_full_prompt=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.sft.enabled=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    trainer.logger='["console"]' \
    trainer.project_name='fixEngineMetrics' \
    trainer.experiment_name="$branch/ppo_micro_batch_size_per_gpu$ppo_micro_batch_size_per_gpu" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=40 \
    trainer.use_legacy_worker_impl=disable \
    trainer.total_epochs=2 \
    trainer.total_training_steps=10 \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
