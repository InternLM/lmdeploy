# Installation

LMDeploy is a python library for compressing, deploying, and serving Large Language Models(LLMs) and Vision-Language Models(VLMs).
Its core inference engines include TurboMind Engine and PyTorch Engine. The former is developed by C++ and CUDA, striving for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

It supports LLMs and VLMs deployment on both Linux and Windows platforms. Furthermore, it is compatible with the following NVIDIA GPUs:

- Volta(sm70): V100
- Turing(sm75): 20 series, T4
- Ampere(sm80,sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace(sm89): 40 series

## Install with pip (Recommend)

It is recommended installing lmdeploy using pip in a conda environment (python 3.10 - 3.13):

```shell
conda create -n lmdeploy python=3.12 -y
conda activate lmdeploy
pip install lmdeploy
```

## FlashKDA in CUDA 13 Docker builds

Images built from `docker/Dockerfile` or `docker/Dockerfile_dev` with
`--build-arg CUDA_VERSION=cu130` include a pinned [FlashKDA](https://github.com/MoonshotAI/FlashKDA)
wheel. CUDA 12.8 builds retain the Triton KDA implementation.

FlashKDA is disabled by default because FLA 0.5.2 does not check GPU architecture
before selecting it. On supported SM90+ GPUs, such as H100 or H200, pass
`-e FLA_FLASH_KDA=1` to `docker run` to enable it for eligible KDA prefill calls.
Keep `FLA_DISABLE_BACKEND_DISPATCH` unset. Set `FLA_FLASH_KDA=0` to use Triton again;
restart the service after changing these variables. KDA decode is unchanged.

The GLM-5.3 KDA prefill adapter already supplies FlashKDA's required options.
Verify dispatch with a prefill trace: FlashKDA emits `_flash_kda_fwd_prepare` and
`_flash_kda_fwd_recurrence` kernels. Compare prefill outputs and recurrent states,
and validate model quality before enabling it for production traffic.

## Install from source

By default, LMDeploy will build with NVIDIA CUDA support, utilizing both the Turbomind and PyTorch backends. TurboMind requires C++20 for both host C++ and CUDA compilation, CMake 3.25.2 or newer, and CUDA Toolkit 12.0 or newer. Use a C++20-capable host compiler supported by your CUDA Toolkit, such as GCC 10 or newer on Linux or Visual Studio 2022 on Windows.

Once the CUDA toolkit is successfully set up, you can build and install LMDeploy with a single command:

```shell
pip install git+https://github.com/InternLM/lmdeploy.git
```

You can also explicitly disable the Turbomind backend to avoid CUDA compilation by setting the `DISABLE_TURBOMIND` environment variable:

```shell
DISABLE_TURBOMIND=1 pip install git+https://github.com/InternLM/lmdeploy.git
```

If you prefer a specific version instead of the `main` branch of LMDeploy, you can specify it in your command:

```shell
pip install https://github.com/InternLM/lmdeploy/archive/refs/tags/v0.11.0.zip
```

If you want to build LMDeploy with support for Ascend, Cambricon, or MACA, install LMDeploy with the corresponding `LMDEPLOY_TARGET_DEVICE` environment variable.

LMDeploy also supports installation on AMD GPUs with ROCm.

```shell
#The recommended way is to use the official ROCm PyTorch Docker image with pre-installed dependencies:
docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --network=host \
    --shm-size 32G \
    -v /root:/workspace \
    rocm/pytorch:latest


#Once inside the container, install LMDeploy with ROCm support:
LMDEPLOY_TARGET_DEVICE=rocm pip install  git+https://github.com/InternLM/lmdeploy.git
```
