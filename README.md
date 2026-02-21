# Activate conda in script

```bash
eval "$(conda shell.bash hook)"
```

# Install npm

```bash
https://stackoverflow.com/a/31046037/10965084

```

# Install codex 

```bash
npm i -g @openai/codex
```

# Install claude

```bash
npm install -g @anthropic-ai/claude-code
```

Login to claude: console.anthropic on tamu google account

# Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Env setup

```bash

pip install --no-deps -e . # add verl to root w/o installing dependencies, which may cause conflicts with other packages in the environment. You can install dependencies separately if needed.

/nvme-data/jacob/verl$ pytest -v # run tests to verify the installation
```
In some cases (server 9) the following need to be done after vanilla env init:

```bash
python -m pip install antlr4-python3-runtime==4.9.3
python -m pip install transformers==4.57.0 
python -m pip install numpy==2.2
```


# Debug args

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    trainer.resume_mode=disable \
    data.train_files=$DATA_PATH/gsm8k/train.parquet \
    data.val_files=$DATA_PATH/gsm8k/test.parquet \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5 \
    data.train_batch_size=2 \
    trainer.logger='["console"]' \
```

# Preprocess data

```bash
DATA_DIR=$(pwd)
export HF_HOME=$DATA_DIR

python -m examples.data_preprocess.gsm8k --local_save_dir $DATA_DIR/gsm8k
```

Pre-commit:

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
git commit -m "your commit message"
```

# FSDP env setup 

```bash
#!/bin/bash

export MAX_JOBS=32

conda create -n verl python=3.12 -y
conda activate verl

uv pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    uv pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

uv pip install --no-cache-dir flashinfer-python==0.3.1


echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"



echo "Successfully installed all packages"

```

# Megatron + TE + Bridge install (WORKING SOLUTION on dive8)

Use this approach instead of the minimal one at the top. This works because:
1. vllm installs PyTorch with correct CUDA version
2. cuDNN/NCCL from PyPI wheels avoid system CUDA mismatch
3. Environment variables guide builds to use the right libraries
4. Install megatron-bridge from GitHub (not PyPI) to get VLMLoRA support

## Key Issues Solved

This installation handles several common problems on HPC systems:
- **CUDA version mismatch**: System has CUDA 13.1, but PyTorch uses 12.8
- **Missing cuDNN headers**: Compiler can't find cudnn.h from PyTorch's bundled CUDA
- **VLMLoRA not available**: PyPI version (0.2.0rc6) lacks VLMLoRA, need 0.3.0+ from GitHub
- **Optional deps fail to build**: Skip mamba-ssm, causal-conv1d which require CUDA compilation

## Requirements

- Python 3.12
- NVIDIA GPU with driver supporting CUDA 12.4+
- No sudo privileges required
- Conda/Mamba and uv package manager

## Installation Steps

```bash

conda create -n verlMega python=3.12 -y
conda activate verlMega

echo "1. install vllm (brings PyTorch with correct CUDA version)"
uv pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    uv pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# python -m pip install --no-cache-dir flashinfer-python==0.3.1



echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"



uv pip install gpustat ipykernel ipython


export_variable_if_exists() {
  local VAR_NAME="$1"
  local PATH_VALUE="$2"
  local APPEND="${3:-false}"

  if [ -d "$PATH_VALUE" ] || [ -f "$PATH_VALUE" ]; then
    if [ "$APPEND" = "true" ]; then
      if [ -n "${!VAR_NAME:-}" ]; then
        export "$VAR_NAME"="${!VAR_NAME}:$PATH_VALUE"
      else
        export "$VAR_NAME"="$PATH_VALUE"
      fi
    else
      export "$VAR_NAME"="$PATH_VALUE"
    fi
    echo "$VAR_NAME is set to $PATH_VALUE"
  else
    echo "ERROR: $VAR_NAME path not found at $PATH_VALUE" >&2
    # exit 1
  fi
}

# -----------------------------------------------------------------------------
# TransformerEngine (no-sudo HPC friendly): install NCCL/cuDNN wheels + export paths
# -----------------------------------------------------------------------------
echo "6. install TE deps (NCCL + cuDNN) from PyPI"
uv pip install --no-cache-dir "nvidia-nccl-cu12" "nvidia-cudnn-cu12"

# Compute roots inside this env
NCCL_HOME="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "nccl")
PY
)"
CUDNN_HOME="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "cudnn")
PY
)"

# Export paths (headers + libs)
export_variable_if_exists NCCL_HOME "$NCCL_HOME" false
export_variable_if_exists CPLUS_INCLUDE_PATH "$NCCL_HOME/include" true
export_variable_if_exists C_INCLUDE_PATH      "$NCCL_HOME/include" true

if [ -d "$NCCL_HOME/lib" ]; then
  export_variable_if_exists LIBRARY_PATH    "$NCCL_HOME/lib" true
  export_variable_if_exists LD_LIBRARY_PATH "$NCCL_HOME/lib" true
elif [ -d "$NCCL_HOME/lib64" ]; then
  export_variable_if_exists LIBRARY_PATH    "$NCCL_HOME/lib64" true
  export_variable_if_exists LD_LIBRARY_PATH "$NCCL_HOME/lib64" true
else
  echo "ERROR: Neither $NCCL_HOME/lib nor $NCCL_HOME/lib64 exists" >&2
#   exit 1
fi

export_variable_if_exists CUDNN_HOME "$CUDNN_HOME" false
export_variable_if_exists CPLUS_INCLUDE_PATH "$CUDNN_HOME/include" true
export_variable_if_exists C_INCLUDE_PATH      "$CUDNN_HOME/include" true
if [ -d "$CUDNN_HOME/lib" ]; then
  export_variable_if_exists LD_LIBRARY_PATH "$CUDNN_HOME/lib" true
elif [ -d "$CUDNN_HOME/lib64" ]; then
  export_variable_if_exists LD_LIBRARY_PATH "$CUDNN_HOME/lib64" true
else
  echo "ERROR: Neither $CUDNN_HOME/lib nor $CUDNN_HOME/lib64 exists" >&2
  exit 1
fi

# Add all NVIDIA CUDA library paths (cublas, cuda_runtime, etc.)
echo "Adding NVIDIA CUDA library paths to LD_LIBRARY_PATH"
NVIDIA_ROOT="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia")
PY
)"
for lib_dir in "$NVIDIA_ROOT"/*/lib; do
  if [ -d "$lib_dir" ]; then
    export_variable_if_exists LD_LIBRARY_PATH "$lib_dir" true
  fi
done

# Optional: helps some CMake find logic
export_variable_if_exists CMAKE_PREFIX_PATH "$NCCL_HOME" true
export_variable_if_exists CMAKE_PREFIX_PATH "$CUDNN_HOME" true

# Sanity checks
test -f "$NCCL_HOME/include/nccl.h" || (echo "Missing nccl.h at $NCCL_HOME/include/nccl.h" >&2; exit 1)
python - <<'PY'
import site, pathlib, glob
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "cudnn"
hits = glob.glob(str(root / "**/libcudnn_graph.so*"), recursive=True)
print("Found libcudnn_graph:", hits[:3])
assert hits, "libcudnn_graph not found in nvidia-cudnn-cu12 install"
PY

echo "7. install TransformerEngine"
uv pip install --no-build-isolation transformer_engine[pytorch]

echo "8. verify import"
python - <<'PY'
import transformer_engine as te
print("TransformerEngine import OK:", te.__version__)
PY

uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1

echo "9. install Megatron-Bridge (from GitHub to get VLMLoRA support)"
# Install megatron-bridge 0.3.0+ from GitHub (PyPI version 0.2.0rc6 lacks VLMLoRA)
# Use --no-deps to avoid building causal-conv1d, mamba-ssm which fail with CUDA version mismatches
echo "Installing megatron-bridge 0.3.0rc0+ from GitHub..."
uv pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Install megatron-core without deps
uv pip install --no-deps "megatron-core>=0.15.0"

# Install required runtime dependencies
echo "Installing required dependencies..."
# nvidia-modelopt is required for megatron-bridge 0.3.0+
uv pip install nvidia-modelopt
# Other runtime dependencies (skip dev extras like mamba-ssm, causal-conv1d)
uv pip install apex transformers nltk six importlib-metadata zarr tensorstore packaging

echo "10. Verify megatron-bridge installation"
python - <<'PY'
from megatron.bridge.peft.lora import LoRA, VLMLoRA
print("Megatron-Bridge import OK")
print("LoRA available:", LoRA)
print("VLMLoRA available:", VLMLoRA)
PY

echo "Successfully installed all packages (TransformerEngine + Megatron-Bridge with VLMLoRA) ✅"

```

## Installation Steps on dive6

```bash

conda create -n verlMega python=3.12 -y
conda activate verlMega

echo "1. install vllm (brings PyTorch with correct CUDA version)"
uv pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    uv pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# python -m pip install --no-cache-dir flashinfer-python==0.3.1



echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"



uv pip install gpustat ipykernel ipython


export_variable_if_exists() {
  local VAR_NAME="$1"
  local PATH_VALUE="$2"
  local APPEND="${3:-false}"

  if [ -d "$PATH_VALUE" ] || [ -f "$PATH_VALUE" ]; then
    if [ "$APPEND" = "true" ]; then
      if [ -n "${!VAR_NAME:-}" ]; then
        export "$VAR_NAME"="${!VAR_NAME}:$PATH_VALUE"
      else
        export "$VAR_NAME"="$PATH_VALUE"
      fi
    else
      export "$VAR_NAME"="$PATH_VALUE"
    fi
    echo "$VAR_NAME is set to $PATH_VALUE"
  else
    echo "ERROR: $VAR_NAME path not found at $PATH_VALUE" >&2
    # exit 1
  fi
}

# -----------------------------------------------------------------------------
# dive6 only: Install CUDA toolkit headers and set up NVTX paths
# -----------------------------------------------------------------------------
# Required for TransformerEngine compilation on dive6 systems
# The pip-installed cuda-runtime package is missing critical CUDA headers like crt/host_defines.h
# This causes compilation errors: "fatal error: crt/host_defines.h: No such file or directory"
echo "5.5. [dive6 only] install CUDA toolkit for complete headers"
conda install -c nvidia cuda-toolkit=12.4 -y
# Alternative: pip install nvidia-cuda-nvcc-cu12
# If conda install fails, uncomment the line below:
# uv pip install nvidia-cuda-nvcc-cu12

# [dive6 only] Add NVTX include path for TransformerEngine compilation
# TransformerEngine needs nvtx3/nvToolsExt.h which is in the pip-installed nvidia packages
NVTX_INCLUDE="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "nvtx" / "include")
PY
)"
export_variable_if_exists CPLUS_INCLUDE_PATH "$NVTX_INCLUDE" true
export_variable_if_exists C_INCLUDE_PATH "$NVTX_INCLUDE" true

# [dive6 only] Fix TORCH_CUDA_ARCH_LIST for RTX A6000 (compute capability 8.6)
# The default list includes non-existent architectures (11.0, 12.0) which cause compilation errors
# Error: "ValueError: Unknown CUDA arch (11.0) or GPU not supported"
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0"
echo "TORCH_CUDA_ARCH_LIST set to: $TORCH_CUDA_ARCH_LIST"

# -----------------------------------------------------------------------------
# TransformerEngine (no-sudo HPC friendly): install NCCL/cuDNN wheels + export paths
# -----------------------------------------------------------------------------
echo "6. install TE deps (NCCL + cuDNN) from PyPI"
uv pip install --no-cache-dir "nvidia-nccl-cu12" "nvidia-cudnn-cu12"

# Compute roots inside this env
NCCL_HOME="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "nccl")
PY
)"
CUDNN_HOME="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "cudnn")
PY
)"

# Export paths (headers + libs)
export_variable_if_exists NCCL_HOME "$NCCL_HOME" false
export_variable_if_exists CPLUS_INCLUDE_PATH "$NCCL_HOME/include" true
export_variable_if_exists C_INCLUDE_PATH      "$NCCL_HOME/include" true

if [ -d "$NCCL_HOME/lib" ]; then
  export_variable_if_exists LIBRARY_PATH    "$NCCL_HOME/lib" true
  export_variable_if_exists LD_LIBRARY_PATH "$NCCL_HOME/lib" true
elif [ -d "$NCCL_HOME/lib64" ]; then
  export_variable_if_exists LIBRARY_PATH    "$NCCL_HOME/lib64" true
  export_variable_if_exists LD_LIBRARY_PATH "$NCCL_HOME/lib64" true
else
  echo "ERROR: Neither $NCCL_HOME/lib nor $NCCL_HOME/lib64 exists" >&2
#   exit 1
fi

export_variable_if_exists CUDNN_HOME "$CUDNN_HOME" false
export_variable_if_exists CPLUS_INCLUDE_PATH "$CUDNN_HOME/include" true
export_variable_if_exists C_INCLUDE_PATH      "$CUDNN_HOME/include" true
if [ -d "$CUDNN_HOME/lib" ]; then
  export_variable_if_exists LD_LIBRARY_PATH "$CUDNN_HOME/lib" true
elif [ -d "$CUDNN_HOME/lib64" ]; then
  export_variable_if_exists LD_LIBRARY_PATH "$CUDNN_HOME/lib64" true
else
  echo "ERROR: Neither $CUDNN_HOME/lib nor $CUDNN_HOME/lib64 exists" >&2
  exit 1
fi

# Add all NVIDIA CUDA library paths (cublas, cuda_runtime, etc.)
echo "Adding NVIDIA CUDA library paths to LD_LIBRARY_PATH"
NVIDIA_ROOT="$(python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "nvidia")
PY
)"
for lib_dir in "$NVIDIA_ROOT"/*/lib; do
  if [ -d "$lib_dir" ]; then
    export_variable_if_exists LD_LIBRARY_PATH "$lib_dir" true
  fi
done

# Optional: helps some CMake find logic
export_variable_if_exists CMAKE_PREFIX_PATH "$NCCL_HOME" true
export_variable_if_exists CMAKE_PREFIX_PATH "$CUDNN_HOME" true

# Sanity checks
test -f "$NCCL_HOME/include/nccl.h" || (echo "Missing nccl.h at $NCCL_HOME/include/nccl.h" >&2; exit 1)
python - <<'PY'
import site, pathlib, glob
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "cudnn"
hits = glob.glob(str(root / "**/libcudnn_graph.so*"), recursive=True)
print("Found libcudnn_graph:", hits[:3])
assert hits, "libcudnn_graph not found in nvidia-cudnn-cu12 install"
PY

echo "7. install TransformerEngine"
uv pip install --no-build-isolation transformer_engine[pytorch]

echo "8. verify import"
python - <<'PY'
import transformer_engine as te
print("TransformerEngine import OK:", te.__version__)
PY

uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1

echo "9. install Megatron-Bridge (from GitHub to get VLMLoRA support)"
# Install megatron-bridge from GitHub (PyPI version 0.2.0rc6 lacks VLMLoRA)
# Use --no-deps to avoid building causal-conv1d, mamba-ssm which fail with CUDA version mismatches
# Pin to commit 953aabf for compatibility with megatron-core 0.15.x
echo "Installing megatron-bridge from GitHub..."
uv pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@953aabf75c0500180dc14a6a76cf9e7e7c4baec7

# Install megatron-core without deps
uv pip install --no-deps "megatron-core>=0.15.0"

# Install required runtime dependencies
echo "Installing required dependencies..."
# nvidia-modelopt is required for megatron-bridge 0.3.0+
uv pip install nvidia-modelopt
# Other runtime dependencies (skip dev extras like mamba-ssm, causal-conv1d)
uv pip install apex transformers nltk six importlib-metadata zarr tensorstore packaging


uv pip install transformers==4.57.6
python -m pip install "numpy<2"

echo "10. Verify megatron-bridge installation"
python - <<'PY'
from megatron.bridge.peft.lora import LoRA, VLMLoRA
print("Megatron-Bridge import OK")
print("LoRA available:", LoRA)
print("VLMLoRA available:", VLMLoRA)
PY

echo "Successfully installed all packages (TransformerEngine + Megatron-Bridge with VLMLoRA) ✅"

```

# Megatron train script

```bash
#!/usr/bin/env bash
set -xeuo pipefail

# Need to install Megatron-Bridge
# NOTE: Make sure you use Megatron-Bridge later than 0.2.0 
# (Recommend https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/953aabf75c0500180dc14a6a76cf9e7e7c4baec7 or later)
# for proper MoE LoRA support.

# For Megatron communication/computation overlapping
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_VISIBLE_DEVICES=8,9
export NCCL_P2P_DISABLE=1
PROJECT_PATH=$PWD
DATA_PATH=$PROJECT_PATH/../verlData

############################ Quick Config ############################

rollout_name="vllm" # sglang or vllm
project_name='verl_grpo_example_gsm8k_math'
exp_name='qwen2_7b_megatron_lora'

adv_estimator=grpo

max_prompt_length=1024
max_response_length=1024
train_prompt_bsz=2

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.train_batch_size=$train_prompt_bsz
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
    actor_rollout_ref.model.lora.rank=64
    actor_rollout_ref.model.lora.alpha=32
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    # # Optional: Use canonical LoRA
    # actor_rollout_ref.model.lora.type="canonical_lora"
    # actor_rollout_ref.model.lora.target_modules='["linear_q","linear_k","linear_v","linear_proj","linear_fc1_up","linear_fc1_gate","linear_fc2"]'

    # # Optional: Add dropout to LoRA layers
    # actor_rollout_ref.model.lora.dropout=0.05
    # actor_rollout_ref.model.lora.dropout_position=pre
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=4
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

ALGORITHM=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
)

############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"


```

# FSDP train script

```bash
set -x

set -x

eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

MODEL=Qwen/Qwen2.5-0.5B-Instruct

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

TRAIN_FILES="['$gsm8k_train_path']"
TEST_FILES="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.train_batch_size=2 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='deepseek_llm_7b_function_rm_math' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

```

# Doc string instructions

```md
# Docstring Instructions for On-Policy Distillation

## Overview
Add comprehensive docstrings to all functions and classes in the on-policy distillation PR, following the format from `compute_policy_loss_vanilla` in `verl/trainer/ppo/core_algos.py:1160`.

This includes:
- Adding docstrings to new functions/classes that lack them
- Fixing existing docstrings that are incomplete, inaccurate, or don't follow the standard format
- Ensuring all tensor parameters include shape information
- Fixing any bugs discovered in the code (e.g., undefined variables)

## Finding What Needs Documentation

### Step 1: Identify Files Changed in the PR

Use git diff to find all Python files modified in your branch:

```bash
# See all changed files compared to main branch
git diff --name-only main...HEAD | grep "\.py$"

# Or if you want to see which functions were added/modified
git diff main...HEAD --stat
```


### Step 2: Find New or Modified Functions

Use git diff to see the actual changes and identify new functions:

```bash
# See full diff of a specific file
git diff main...HEAD verl/trainer/distillation/losses.py

# Filter to show only function definitions added
git diff main...HEAD verl/trainer/distillation/losses.py | grep "^+def "
```

Look for:
- Lines starting with `+def` (new functions)
- Lines starting with `+class` (new classes)
- Existing functions with modified signatures or docstrings

### Step 3: Review Each Function/Class

For each function or class you find, check:

1. **Does it have a docstring at all?**
   - Look for TODO placeholders
   - Look for single-line summaries without Args/Returns
   - Look for completely missing docstrings

2. **Is the existing docstring complete?**
   - Does it have an Args section for all parameters?
   - Does it have a Returns section?
   - For classes, does it have an Attributes section?

3. **Is the docstring accurate?**
   - Do the parameter names match the function signature?
   - Are shapes documented correctly?
   - Are optional parameters marked as such?

4. **Does it follow the standard format?**
   - Compare against `compute_policy_loss_vanilla` in `verl/trainer/ppo/core_algos.py:1160`
   - Check Args formatting: `param_name (Type):` not `param_name: \`(Type)\``
   - Check for proper indentation

5. **Are there any bugs in the code?**
   - Look for undefined variables
   - Look for parameter mismatches (e.g., using wrong variable names)

### Step 4: Systematic Review Process

For each file, go through line by line:

```bash
# Read a specific file and look at line numbers
cat -n verl/trainer/distillation/losses.py

# Or use grep to find all function definitions
grep -n "^def \|^class " verl/trainer/distillation/losses.py
```

Create a checklist of functions/classes that need work, noting:
- Line number
- Function/class name
- What needs to be fixed (missing docstring, incomplete Args, formatting issues, etc.)

### Common Issues to Look For

1. **TODO placeholders**: Search for `TODO` in docstrings
2. **Single-line docstrings**: Functions with only a summary, no Args/Returns
3. **Duplicate Args entries**: Same parameter documented twice
4. **Missing Returns section**: Functions that return values but don't document them
5. **Incorrect formatting**: Using backticks instead of parentheses for types
6. **Missing shape information**: Tensor parameters without shape docs
7. **Missing optional markers**: Optional params not marked as `(Type, optional)`
8. **Code bugs**: Variables used before definition, mismatched parameter names

### Example Git Diff Analysis

```bash
$ git diff main...HEAD verl/trainer/distillation/losses.py | head -50

# You might see:
+def compute_distillation_loss_kl_estimator(
+    teacher_log_probs: torch.Tensor,
+    student_log_probs: torch.Tensor,
+    ...
+):
+    """Compute the distillation loss and related metrics using KL estimator"""
+    assert config is not None
+    log_p, log_q = clamp_log_probs(log_p, log_q)  # BUG: undefined variables!
```

This reveals:
1. New function `compute_distillation_loss_kl_estimator`
2. Has a single-line docstring (incomplete)
3. Has a bug on line with `clamp_log_probs` (uses undefined `log_p, log_q`)

## Reference Format

The standard docstring format used in this codebase (see `compute_policy_loss_vanilla` at verl/trainer/ppo/core_algos.py:1160):

```python
def function_name(param1: Type1, param2: Type2, ...) -> ReturnType:
    """
    One-line summary of what the function does.

    (Optional) Additional context, source references, or background information.
    Can span multiple lines if needed.

    Args:
        param1 (Type1):
            Description of param1, including shape information if applicable.
        param2 (Type2):
            Description of param2, including shape information if applicable.
        optional_param (Type, optional):
            Description of optional parameter. Defaults to value.
        config: `(ConfigType)`:
            Description of config parameter.

    Returns:
        ReturnType: Description of return value, including shape information if applicable.
        For tuple returns: tuple[Type1, Type2]: Description of tuple elements.
    """
```

### Key Format Requirements

1. **Summary**: One clear line describing the function's purpose
2. **Additional context** (optional): References, formulas, background info
3. **Args section**:
   - Format: `param_name (Type):` or `param_name (Type, optional):`
   - Indent description under the parameter
   - Include tensor shapes: `shape (batch_size, seq_length, vocab_size)`
   - Note default values for optional parameters
   - Config parameters can use backtick style: `config: \`(ConfigType)\`:`
4. **Returns section**:
   - Format: `ReturnType: Description`
   - For tuples, list each element with indentation
   - Include shape information for tensors
5. **Attributes section** (for classes):
   - Format: `attribute_name (Type):`
   - Describe purpose and any computed/derived attributes

## Examples

### Good Example: Function with Tensor Parameters

```python
def topk_logprobs_from_logits(
    logits: torch.Tensor, k: int, compute_topk: bool, gather_topk: bool, topk_indices: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute and/or gather top-k log probabilities from logits.

    This function supports two modes:
    1. Computing new top-k log probabilities from logits
    2. Gathering log probabilities at pre-specified indices
    Both modes can be combined to gather from both teacher and student top-k indices.

    Args:
        logits (torch.Tensor):
            Logits from model forward pass, shape (*, vocab_size).
        k (int):
            Number of top log probabilities to compute or gather.
        compute_topk (bool):
            Whether to compute top-k log probabilities from the logits.
        gather_topk (bool):
            Whether to gather log probabilities at indices specified by topk_indices.
        topk_indices (torch.Tensor, optional):
            Pre-computed indices for gathering log probabilities, shape (*, k) or (*, 2*k).
            Required when gather_topk is True. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - topk_logprobs: Top-k log probabilities, shape (*, k) or (*, 2*k).
            - topk_indices: Indices corresponding to the top-k log probabilities, same shape as topk_logprobs.
    """
```

### Good Example: Class with Attributes

```python
@dataclass
class DistillationLossInfo:
    """
    Information about a distillation loss function to be registered.

    Attributes:
        names (list[str] | str):
            Name(s) of the distillation loss function. Can be a single name or list of aliases.
        use_student_topk (bool):
            Whether the loss function requires student top-k log probabilities. Defaults to False.
        use_teacher_topk (bool):
            Whether the loss function requires teacher top-k log probabilities. Defaults to False.
        use_full (bool):
            Whether the loss function requires full vocabulary log probabilities. Defaults to False.
        use_topk (bool):
            Computed attribute indicating whether any top-k log probabilities are needed.
            Set automatically in __post_init__ based on use_student_topk or use_teacher_topk.
    """
    names: list[str] | str
    use_student_topk: bool = False
    use_teacher_topk: bool = False
    use_full: bool = False
```

## Tips

- Always include shape information for tensor parameters
- Use `(*, dim)` notation when shapes are flexible
- Document what each stage does in functions that handle multiple stages
- Explain formulas and algorithms when relevant
- Add references to papers or external documentation when applicable
- For config parameters, use the backtick style: `config: \`(ConfigType)\``
- Mark optional parameters and include their default values
- For tuple returns, list each element on a separate indented line

## Fixing Existing Docstrings

Not all docstrings need to be written from scratch. Many functions already have partial or incorrectly formatted docstrings that need to be fixed:

### Common Fixes Needed

1. **Formatting Fixes**
   ```python
   # WRONG:
   Args:
       name: `(str)`
           The name of the loss function.

   # CORRECT:
   Args:
       name (str):
           The name of the loss function.
   ```

2. **Missing Returns Section**
   ```python
   # INCOMPLETE:
   def register_loss(info):
       """Register a loss function.

       Args:
           info: Loss information.
       """

   # COMPLETE:
   def register_loss(info):
       """Register a loss function.

       Args:
           info: Loss information.

       Returns:
           function: Decorator function that registers the loss.
       """
   ```

3. **Duplicate Args Entries**
   ```python
   # WRONG (response_mask listed twice):
   Args:
       response_mask (torch.Tensor):
           Mask for tokens.
       config (Config):
           Configuration.
       response_mask (torch.Tensor):
           Mask for tokens.

   # CORRECT:
   Args:
       response_mask (torch.Tensor):
           Mask for tokens.
       config (Config):
           Configuration.
   ```

4. **Missing Shape Information**
   ```python
   # INCOMPLETE:
   Args:
       logits (torch.Tensor):
           Model logits.

   # COMPLETE:
   Args:
       logits (torch.Tensor):
           Model logits, shape (batch_size, sequence_length, vocab_size).
   ```

5. **Inaccurate Descriptions**
   - Parameter names in docstring don't match actual function parameters
   - Descriptions don't accurately reflect what the code does
   - Missing information about special behavior (e.g., duplicate handling)

### Fixing Code Bugs Found During Documentation

While documenting, you may discover bugs in the code itself:

```python
# BUG: Using undefined variables
def compute_loss(teacher_log_probs, student_log_probs):
    """Compute loss..."""
    log_p, log_q = clamp_log_probs(log_p, log_q)  # log_p and log_q undefined!

# FIX: Use the actual parameter names
def compute_loss(teacher_log_probs, student_log_probs):
    """Compute loss..."""
    log_p, log_q = clamp_log_probs(teacher_log_probs, student_log_probs)
```

**Always fix these bugs when you find them** - documenting broken code doesn't help anyone!


```

<div align="center">
 👋 Hi, everyone!
    verl is a RL training library initiated by <b>ByteDance Seed team</b> and maintained by the verl community.
    <br>
    <br>
</div>

<div align="center">

<a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verl-project/shared_invite/zt-3c6mc2khw-v0lo6NfDPuFP6OnkrZwfqw"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLMs</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex post-training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Ready integration with popular HuggingFace models

verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

<div align="center">
 <img src="https://github.com/verl-project/verl-data/blob/main/images/verl-arch.png?raw=true" width="400" alt="verl-arch.png">
</div>

</p>

## News

- [2026/01] verl has been migrated to the [verl-project](https://github.com/verl-project)
- [2026/01] verl first meetup was successfully held in Shanghai on 01/10, hosted by Volcengine and NVIDIA, the slides has been uploaded to [verl-data](https://github.com/verl-project/verl-data).
- [2026/01] The `recipe` directory has been migrated to a dedicated repository: [verl-recipe](https://github.com/verl-project/verl-recipe) and added as a submodule. See https://github.com/volcengine/verl/pull/4795. It can be used as it was after `git submodule update --init --recursive recipe`. Note that [`transfer_queue`](verl/experimental/transfer_queue), [`fully_async_policy`](verl/experimental/fully_async_policy), [`one_step_off_policy`](verl/experimental/one_step_off_policy) and [`vla`](verl/experimental/vla) are kept under [`verl/experimental`](verl/experimental) since they are planned to be merged into the main library. Use them through `verl.experimental.{module}`.
- [2025/12] [Mind Lab](https://macaron.im/mindlab) successfully used [verl](https://github.com/volcengine/verl) and [Megatron-bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) to train GRPO Lora for Trillion-parameter model on 64 H800 - See their [techblog](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus).
- [2025/10] verl is presented in the [PyTorch Conference 2025](https://pytorch.org/event/pytorch-conference-2025/).
- [2025/08] verl is presented in the [PyTorch Expert Exchange Webinar](https://www.youtube.com/watch?v=Vd79NmmqY3Q&t=2s). [Slides](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl_talk_pytorch_2025_08.pdf) available.
- [2025/07] The [ReTool](https://arxiv.org/pdf/2504.11536) recipe is fully open sourced. [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
- [2025/07] The first verl meetup will be held at ICML Vancouver on July 16th! Please [join us](https://lu.ma/0ek2nyao) if you are at ICML! (onsite only)
- [2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671B and Qwen3-235B](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
- [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
<details><summary> more... </summary>
<ul>
  <li>[2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.</li>
  <li>[2025/07] verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.</li>
  <li>[2025/06] verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!</li>
  <li> [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.</li>
  <li>[2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.</li>
  <li>[2025/04] We will give a tutorial about latest post-training techniques and programming guide for verl at [ICLR 2025 Expo](https://iclr.cc/virtual/2025/calendar?filter_events=Expo+Talk+Panel&filter_rooms=), [SCI-FM workshop](https://open-foundation-model.github.io/) and [LMSys afterparty](https://lu.ma/d23nyynm). Talk materials available [here](https://github.com/eric-haibin-lin/verl-community/tree/main/iclr25). </li>
  <li>[2025/03] verl v0.3.0.post1 is released! See [release note](https://github.com/volcengine/verl/releases/) for details. It achieves [~1.4x speedup](https://tongyx361.github.io/blogs/posts/verl-intro/#/verl-flexible-and-efficient-rl-for-llms) compared to prev versions.</li>
  <li>[2025/05] verl will be presented at [A2M Shanghai](https://a2m.msup.com.cn/home/?aid=4488&city=shanghai) on 5/16 - 5/17.</li>
  <li>[2025/05] verl will be presented at [GOSIM x PyTorch Day 2025](https://paris2025.gosim.org/). See you in Paris! </li>
  <li>[2025/03] We introduced the programming model of verl at the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) and [verl intro and updates](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf) at the [SGLang-LMSYS Org Meetup](https://lu.ma/ntjrr7ig) in Sunnyvale mid-March.</li>
  <li>[2025/03] We will present verl(HybridFlow) at EuroSys 2025. See you in Rotterdam!</li>
  <li>[2025/02] verl v0.2.0.post2 is released!</li>
  <li>[2025/02] We presented verl in the <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray Meetup</a>. See you in San Jose!</li>
  <li>[2025/01] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM. The RL scaling preview model is trained using verl, reaching OpenAI O1-level performance on math benchmarks (70.0 pass@1 on AIME).</li>
  <li>[2024/12] verl is presented at Ray Forward 2024. Slides available <a href="https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf">here</a></li>
  <li>[2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. <a href="https://github.com/eric-haibin-lin/verl-data/tree/neurips">Slides</a> and <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">video</a> available.</li>
  <li>[2024/10] verl is presented at Ray Summit. <a href="https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37">Youtube video</a> available.</li>
  <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>
</details>

## Key Features

- **FSDP**, **FSDP2** and **Megatron-LM** for training.
- **vLLM**, **SGLang** and **HF Transformers** for rollout generation.
- Compatible with Hugging Face Transformers and Modelscope Hub: [Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh), Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
- Supervised fine-tuning.
- Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [GSPO](https://github.com/verl-project/verl-recipe/tree/main/gspo/), [ReMax](examples/remax_trainer/), [REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](https://github.com/verl-project/verl-recipe/tree/main/prime/), [DAPO](https://github.com/verl-project/verl-recipe/tree/main/dapo/), [DrGRPO](https://github.com/verl-project/verl-recipe/tree/main/drgrpo), [KL_Cov & Clip_Cov](https://github.com/verl-project/verl-recipe/tree/main/entropy) etc.
  - Support model-based reward and function-based reward (verifiable reward) for math, [coding](https://github.com/volcengine/verl-recipe/tree/main/dapo), etc
  - Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh) with Qwen2.5-vl, Kimi-VL
  - [Multi-turn with tool calling](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
- LLM alignment recipes such as [Self-play preference optimization (SPPO)](https://github.com/verl-project/verl-recipe/tree/main/sppo)
- Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
- Scales up to 671B models and hundreds of GPUs with [expert parallelism](https://github.com/volcengine/verl/pull/1467)
- Multi-gpu [LoRA RL](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) support to save memory.
- Experiment tracking with wandb, swanlab, mlflow and tensorboard.
- Hardware Support: Supports NVIDIA, AMD, [Ascend](https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_quick_start.rst)

## Upcoming Features and Changes

- Q3 Roadmap https://github.com/volcengine/verl/issues/2388
- DeepSeek 671b optimizations with Megatron https://github.com/volcengine/verl/issues/1033
- Multi-turn rollout and tools using optimizations https://github.com/volcengine/verl/issues/1882
- [Agent integration](https://github.com/volcengine/verl/tree/main/verl/experimental/agent_loop)
- Async and off-policy architecture https://github.com/volcengine/verl/pull/2231
- List of breaking changes since v0.4 https://github.com/volcengine/verl/discussions/2270

## Getting Started

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

**Quickstart:**

- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
- [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
- [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

**Running a PPO example step-by-step:**

- [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
- [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
- [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

**Reproducible algorithm baselines:**

- [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

**For code explanation and advance usage (extension):**

- PPO Trainer and Workers

  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)

- Advanced Usage and Extension
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
  - [Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
  - [Sandbox Fusion Integration](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)

**Blogs from the community**

- [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
- [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
- [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
- [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
- [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## Performance Tuning Guide

The performance is essential for on-policy RL algorithm. We have written a detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) to help you optimize performance.

## Upgrade to vLLM >= v0.8.2

verl now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information. Please avoid vllm 0.7.x, which contains bugs that may lead to OOMs and unexpected errors.

## Use Latest SGLang

SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.

## Upgrade to FSDP2

verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:

```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

## AMD Support (ROCm Kernel)

verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.

## Citation and acknowledgement

If you find the project helpful, please cite:

- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## Awesome Projects Built with `verl`

Welcome to register your awesome project build with `verl` for other developers' reference!

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tuning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): [A no human curated data self-play framework for reasoning](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent): MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
- [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS): A Post-training recipe for scaling RL on Advanced Reasoning models ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
- [PACS](https://github.com/ritzz-ai/PACS): Implicit Actor Critic Coupling via a Supervised Learning Framework for RLVR ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/PACS)
- [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
- [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
- [Agent Lightning](https://github.com/microsoft/agent-lightning): A flexible and extensible framework that enables seamless agent optimization for any existing agent framework. ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)
- [VTool-R1](https://github.com/VTOOL-R1/vtool-r1): VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. ![GitHub Repo stars](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)
- [Kimina-Prover-RL](https://github.com/project-numina/kimina-prover-rl/tree/main/recipe/kimina_prover_rl): Training pipeline for formal theorem proving, based on a paradigm inspired by DeepSeek-R1.
- [RL-PLUS](https://github.com/YihongDong/RL-PLUS): Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization.
- [rStar2-Agent](https://github.com/microsoft/rStar): Using reinforcement learning with multi-step tool-calling for math tasks, rStar2-Agent-14B reaches frontier-level math reasoning in just 510 RL training steps ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/rStar)
- [Vision-SR1](https://github.com/zli12321/Vision-SR1): Self-Rewarding Vision-Language Model via Reasoning Decomposition ![GitHub Repo stars](https://img.shields.io/github/stars/zli12321/Vision-SR1)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL): SimpleVLA-RL: A Simple yet Effective Vision-Language Action Model for Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/SimpleVLA-RL)
- [Table-R1](https://github.com/Table-R1/Table-R1): Table-R1: Inference-Time Scaling for Table Reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/Table-R1/Table-R1)
- [Revisual-R1](https://github.com/CSfufu/Revisual-R1): Revisual-R1: Advancing Multimodal Reasoning From Optimized Cold Start to Staged Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/CSfufu/Revisual-R1)
- [ARES](https://github.com/shawn0728/ARES): ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping ![GitHub Repo stars](https://img.shields.io/github/stars/shawn0728/ARES)
- [Meta-Bandit-LLM](https://github.com/sanxing-chen/meta-bandit-llm): Meta-Bandit-LLM: Long-horizon multiturn interactive training for meta-bandit agents ![GitHub Repo stars](https://img.shields.io/github/stars/sanxing-chen/meta-bandit-llm)
- [PokeeResearch](https://github.com/Pokee-AI/PokeeResearchOSS): PokeeResearch: State-of-the-art 7B DeepResearch Agent that leverages web search and content reading capabilities to answer complex questions using the most up-to-date information available online. ![Github Repo Stars](https://img.shields.io/github/stars/Pokee-AI/PokeeResearchOSS)
- [Search Self-play](https://github.com/Alibaba-Quark/SSP): Pushing the Frontier of Agent Capability without Supervision ![GitHub Repo stars](https://img.shields.io/github/stars/Alibaba-Quark/SSP)
- [OneThinker](https://github.com/tulerfeng/OneThinker): All-in-one Reasoning Model for Image and Video ![GitHub Repo stars](https://img.shields.io/github/stars/tulerfeng/OneThinker)
- [OpenTinker](https://github.com/open-tinker/OpenTinker): Democratizing Agentic Reinforcement Learning as a Service ![GitHub Repo stars](https://img.shields.io/github/stars/open-tinker/OpenTinker)
- [FlowRL](https://github.com/Xuekai-Zhu/FlowRL): Matching reward distributions via **flow balance** for diverse exploration and generalizable reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/Xuekai-Zhu/FlowRL)
- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Seed-Coder](https://github.com/ByteDance-Seed/Seed-Coder): RL training of Seed-Coder boosts performance on competitive programming ![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Seed-Coder)
- [all-hands/openhands-lm-32b-v0.1](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model): A strong, open coding agent model, trained with [multi-turn fine-tuning](https://github.com/volcengine/verl/pull/195)
- [s3](https://github.com/pat-jj/s3) **Efficient Yet Effective** Search Agent Training via RL ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/s3)
- [Rec-R1](https://arxiv.org/pdf/2503.24289): Bridging Generative Large Language Models and Recommendation Systems via Reinforcement Learning
- [Explore RL Data Scaling](https://arxiv.org/abs/2503.22230): Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback
- [FIRE](https://arxiv.org/abs/2410.21236): Flaming-hot initiation with regular execution sampling for large language models
- [DQO](https://arxiv.org/abs/2410.09302): Enhancing multi-Step reasoning abilities of language models through direct Q-function optimization
- [ProRL](https://arxiv.org/abs/2505.24864): Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
- [cognition-engineering](https://github.com/gair-nlp/cognition-engineering): Test time scaling drives cognition engineering. ![GitHub Repo stars](https://img.shields.io/github/stars/gair-nlp/cognition-engineering)
- [Trust Region Preference Approximation](https://github.com/XueruiSu/Trust-Region-Preference-Approximation): A simple and stable **reinforcement learning algorithm** for LLM reasoning. ![GitHub Repo stars](https://img.shields.io/github/stars/XueruiSu/Trust-Region-Preference-Approximation)
- [AdaRFT](https://github.com/uscnlp-lime/verl): Efficient Reinforcement Finetuning via **Adaptive Curriculum Learning** ![GitHub Repo stars](https://img.shields.io/github/stars/uscnlp-lime/verl)
- [critic-rl](https://github.com/HKUNLP/critic-rl): LLM critics for code generation ![GitHub Repo stars](https://img.shields.io/github/stars/HKUNLP/critic-rl)
- [self-rewarding-reasoning-LLM](https://arxiv.org/pdf/2502.19613): self-rewarding and correction with **generative reward models** ![GitHub Repo stars](https://img.shields.io/github/stars/RLHFlow/Self-rewarding-reasoning-LLM)
- [DeepEnlighten](https://github.com/DolbyUUU/DeepEnlighten): Reproduce R1 with **social reasoning** tasks and analyze key findings ![GitHub Repo stars](https://img.shields.io/github/stars/DolbyUUU/DeepEnlighten)
- [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing **3D Spatial Reasoning** in **VLMs** for the **Metaverse** ![GitHub Repo stars](https://img.shields.io/github/stars/PzySeere/MetaSpatial)
- [PURE](https://github.com/CJReinforce/PURE): **Credit assignment** is the key to successful reinforcement fine-tuning using **process reward model** ![GitHub Repo stars](https://img.shields.io/github/stars/CJReinforce/PURE)
- [cognitive-behaviors](https://github.com/kanishkg/cognitive-behaviors): Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs ![GitHub Repo stars](https://img.shields.io/github/stars/kanishkg/cognitive-behaviors)
- [deepscaler](https://github.com/agentica-project/rllm/tree/deepscaler): iterative context scaling with GRPO ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/deepscaler)
- [DAPO](https://dapo-sia.github.io/): the fully open source SOTA RL algorithm that beats DeepSeek-R1-zero-32B ![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)
- [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout): Reinforcing Visual Reasoning with Data Augmentation ![GitHub Repo stars](https://img.shields.io/github/stars/NUS-TRAIL/NoisyRollout)
- [SPEAR](https://github.com/TencentYoutuResearch/SPEAR): **Self-imitation** with **Progressive Exploration** for Agentic Reinforcement Learning (ICLR 2026) ![GitHub Repo stars](https://img.shields.io/github/stars/TencentYoutuResearch/SPEAR)
- [RuleReasoner](https://github.com/bigai-nlco/RuleReasoner): **RuleReasoner:** Reinforced Rule-based Reasoning via **Domain-aware Dynamic Sampling** (ICLR 2026) ![GitHub Repo stars](https://img.shields.io/github/stars/bigai-nlco/RuleReasoner)

## Contribution Guide

See [contributions guide](CONTRIBUTING.md)

## About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society. You can get to know Bytedance Seed better through the following channels👇

<div>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</div>

We are HIRING! Send us an [email](mailto:the.verl.project@gmail.com) if you are interested in internship/FTE opportunities in RL for agents.
