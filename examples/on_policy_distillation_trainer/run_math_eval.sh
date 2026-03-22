



#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

# MODEL="Qwen/Qwen3-1.7B"
# MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL="Qwen/Qwen3-4B"

SAVE_FREQ=20
TEST_FREQ=5
EPOCHS=15

PROJECT_NAME='verl_grpo_example_math'
EXP_NAME="${MODEL}"

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=4096
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH ))

TRAIN_PROMPT_BSZ=600
N_ROLLOUTS_PER_PROMPT=5
PPO_BSZ=600
MICRO_BATCH_SIZE_PER_GPU=2
MAX_TOKEN_LEN_PER_GPU=$(( MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=False

WORLD_SIZE=6

ENFORCE_EAGER=False # true for faster debugging

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k_boxed/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k_boxed/test.parquet

math_train_path=$DATA_PATH/math/train.parquet
math_test_path=$DATA_PATH/math/test.parquet

dapo_test_path=$DATA_PATH/dapo_math_17k/train.parquet

numina_test_path=$DATA_PATH/numina_math_cot_subset_11000/train.parquet

TRAIN_FILES="['$math_train_path']"
TEST_FILES="['$math_test_path', '$gsm8k_test_path', '$dapo_test_path']"
TEST_FILES="['$math_train_path']"
TEST_FILES="['$numina_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path=$MODEL
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=$N_ROLLOUTS_PER_PROMPT
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    reward_model.reward_manager=remote
    custom_reward_function.path=tests/experimental/reward_loop/reward_fn.py
    custom_reward_function.name=compute_score_math_verify
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.val_only=True
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=$SAVE_FREQ
    trainer.test_freq=$TEST_FREQ
    trainer.total_epochs=$EPOCHS
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)



############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"