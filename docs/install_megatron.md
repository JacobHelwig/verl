```bash
conda create -n verlMega python=3.12 -y

conda activate verlMega

# https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/index.html
uv pip install torch --torch-backend=auto
uv pip install --no-build-isolation transformer_engine[pytorch]
uv pip install megatron-bridge
```

Error:

```bash
(verlMega) jacob.a.helwig@csce-dive8:~$ uv pip install torch --torch-backend=auto
uv pip install --no-build-isolation transformer_engine[pytorch]
uv pip install megatron-bridge
Using Python 3.12.12 environment at: miniconda3/envs/verlMega
Resolved 26 packages in 1.18s
Prepared 13 packages in 46.41s
Installed 25 packages in 283ms
 + filelock==3.20.2
 + fsspec==2025.12.0
 + jinja2==3.1.6
 + markupsafe==3.0.3
 + mpmath==1.3.0
 + networkx==3.6.1
 + nvidia-cublas-cu12==12.9.1.4
 + nvidia-cuda-cupti-cu12==12.9.79
 + nvidia-cuda-nvrtc-cu12==12.9.86
 + nvidia-cuda-runtime-cu12==12.9.79
 + nvidia-cudnn-cu12==9.10.2.21
 + nvidia-cufft-cu12==11.4.1.4
 + nvidia-cufile-cu12==1.14.1.1
 + nvidia-curand-cu12==10.3.10.19
 + nvidia-cusolver-cu12==11.7.5.82
 + nvidia-cusparse-cu12==12.5.10.65
 + nvidia-cusparselt-cu12==0.7.1
 + nvidia-nccl-cu12==2.27.5
 + nvidia-nvjitlink-cu12==12.9.86
 + nvidia-nvshmem-cu12==3.3.20
 + nvidia-nvtx-cu12==12.9.79
 + sympy==1.14.0
 + torch==2.9.1+cu129
 + triton==3.5.1
 + typing-extensions==4.15.0
Using Python 3.12.12 environment at: miniconda3/envs/verlMega
Resolved 43 packages in 116ms
Installed 2 packages in 69ms
  × Failed to build `transformer-engine-torch==2.11.0`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)

      [stdout]
      running bdist_wheel
      Guessing wheel URL:
      https://github.com/NVIDIA/TransformerEngine/releases/download/v2.11.0/transformer_engine_torch-2.11.0+cu12torch2.9.1+cu129cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
      Precompiled wheel not found. Building from source...
      running build
      running build_ext
      building 'transformer_engine_torch' extension
      creating build/temp.linux-x86_64-cpython-312/csrc
      creating build/temp.linux-x86_64-cpython-312/csrc/extensions
      creating build/temp.linux-x86_64-cpython-312/csrc/extensions/multi_tensor
      g++ -pthread -B /home/jacob.a.helwig/miniconda3/envs/verlMega/compiler_compat -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2
      -isystem /home/jacob.a.helwig/miniconda3/envs/verlMega/include -fPIC -O2 -isystem /home/jacob.a.helwig/miniconda3/envs/verlMega/include -fPIC
      -I/usr/local/cuda/include -I/home/jacob.a.helwig/.cache/uv/sdists-v9/pypi/transformer-engine-torch/2.11.0/u7ANhD-Wnjn8aUScTlgZR/src/common_headers
      -I/home/jacob.a.helwig/.cache/uv/sdists-v9/pypi/transformer-engine-torch/2.11.0/u7ANhD-Wnjn8aUScTlgZR/src/common_headers/common
      -I/home/jacob.a.helwig/.cache/uv/sdists-v9/pypi/transformer-engine-torch/2.11.0/u7ANhD-Wnjn8aUScTlgZR/src/common_headers/common/include
      -I/home/jacob.a.helwig/.cache/uv/sdists-v9/pypi/transformer-engine-torch/2.11.0/u7ANhD-Wnjn8aUScTlgZR/src/csrc
      -I/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/site-packages/torch/include
      -I/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
      -I/home/jacob.a.helwig/miniconda3/envs/verlMega/include/python3.12 -c csrc/common.cpp -o build/temp.linux-x86_64-cpython-312/csrc/common.o -O3
      -fvisibility=hidden -g0 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=transformer_engine_torch -std=c++17

      [stderr]
      /home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/site-packages/setuptools/_distutils/dist.py:289: UserWarning: Unknown distribution
      option: 'tests_require'
        warnings.warn(msg)
      W0108 11:34:14.454000 1385734 site-packages/torch/utils/cpp_extension.py:630] Attempted to use ninja as the BuildExtension backend but we could not
      find ninja.. Falling back to using the slow distutils backend.
      In file included from /home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/site-packages/torch/include/ATen/cudnn/Handle.h:4,
                       from csrc/common.h:14,
                       from csrc/common.cpp:7:
      /home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/site-packages/torch/include/ATen/cudnn/cudnn-wrapper.h:3:10: fatal error: cudnn.h: No
      such file or directory
          3 | #include <cudnn.h>
            |          ^~~~~~~~~
      compilation terminated.
      error: command '/bin/g++' failed with exit code 1

      hint: This error likely indicates that you need to install a library that provides "cudnn.h" for `transformer-engine-torch@2.11.0`
  help: `transformer-engine-torch` (v2.11.0) was included because `transformer-engine[pytorch]` (v2.11.0) depends on `transformer-engine-torch`
Using Python 3.12.12 environment at: miniconda3/envs/verlMega
Resolved 164 packages in 617ms
      Built antlr4-python3-runtime==4.9.3                                                                                                                      × Failed to build `causal-conv1d==1.5.3.post1`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)

      [stdout]


      torch.__version__  = 2.9.1+cu128


      running bdist_wheel
      Guessing wheel URL:
      https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
      Precompiled wheel not found. Building from source...
      running build
      running build_py
      copying causal_conv1d/__init__.py -> build/lib.linux-x86_64-cpython-312/causal_conv1d
      copying causal_conv1d/cpp_functions.py -> build/lib.linux-x86_64-cpython-312/causal_conv1d
      copying causal_conv1d/causal_conv1d_interface.py -> build/lib.linux-x86_64-cpython-312/causal_conv1d
      copying causal_conv1d/causal_conv1d_varlen.py -> build/lib.linux-x86_64-cpython-312/causal_conv1d
      running build_ext

      [stderr]
      /home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:279: UserWarning: Failed to
      initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
        cpu = _conversion_method_template(device=torch.device("cpu"))
      /home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/dist.py:759: SetuptoolsDeprecationWarning: License
      classifiers are deprecated.
      !!

              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:

              License :: OSI Approved :: BSD License

              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************

      !!
        self._finalize_license_expression()
      W0108 11:34:21.948000 1386458 torch/utils/cpp_extension.py:630] Attempted to use ninja as the BuildExtension backend but we could not find ninja..
      Falling back to using the slow distutils backend.
      Traceback (most recent call last):
        File "<string>", line 334, in run
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 240, in urlretrieve
          with contextlib.closing(urlopen(url, data)) as fp:
                                  ^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 215, in urlopen
          return opener.open(url, data, timeout)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 521, in open
          response = meth(req, response)
                     ^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 630, in http_response
          response = self.parent.error(
                     ^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 559, in error
          return self._call_chain(*args)
                 ^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 492, in _call_chain
          result = func(*args)
                   ^^^^^^^^^^^
        File "/home/jacob.a.helwig/miniconda3/envs/verlMega/lib/python3.12/urllib/request.py", line 639, in http_error_default
          raise HTTPError(req.full_url, code, msg, hdrs, fp)
      urllib.error.HTTPError: HTTP Error 404: Not Found

      During handling of the above exception, another exception occurred:

      Traceback (most recent call last):
        File "<string>", line 11, in <module>
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/build_meta.py", line 432, in build_wheel
          return _build(['bdist_wheel'])
                 ^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/build_meta.py", line 423, in _build
          return self._build_with_temp_dir(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/build_meta.py", line 404, in _build_with_temp_dir
          self.run_setup()
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 354, in <module>
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/__init__.py", line 115, in setup
          return distutils.core.setup(**attrs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/core.py", line 186, in setup
          return run_commands(dist)
                 ^^^^^^^^^^^^^^^^^^
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
          dist.run_commands()
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
          self.run_command(cmd)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "<string>", line 351, in run
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/command/bdist_wheel.py", line 370, in run
          self.run_command("build")
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/command/build.py", line 135, in run
          self.run_command(cmd_name)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/command/build_ext.py", line 96, in run
          _build_ext.run(self)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in
      run
          self.build_extensions()
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 665, in build_extensions
          _check_cuda_version(compiler_name, compiler_version)
        File "/home/jacob.a.helwig/.cache/uv/builds-v0/.tmpijSLK8/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 520, in
      _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
      RuntimeError: ('The detected CUDA version (%s) mismatches the version that was used to compilePyTorch (%s). Please make sure to use the same CUDA
      versions.', '13.1', '12.8')

      hint: This usually indicates a problem with the package or the build environment.
  help: `causal-conv1d` (v1.5.3.post1) was included because `megatron-bridge` (v0.2.0rc6) depends on `megatron-core[dev]` (v0.15.2) which depends on
        `causal-conv1d`
(verlMega) jacob.a.helwig@csce-dive8:~$ 
```

Nvidia

```bash
(verl) jacob.a.helwig@csce-dive8:~/verl$ nvidia-smi
Thu Jan  8 11:35:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:45:00.0 Off |                    0 |
| 30%   58C    P2            299W /  300W |   30566MiB /  46068MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

```

# Megatron + TE + Bridge install (WORKING SOLUTION)

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

echo "11. Install mbridge"
uv pip install mbridge

echo "Successfully installed all packages (TransformerEngine + Megatron-Bridge with VLMLoRA) ✅"

```