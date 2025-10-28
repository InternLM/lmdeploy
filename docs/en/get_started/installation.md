# Installation

LMDeploy is a python library for compressing, deploying, and serving Large Language Models(LLMs) and Vision-Language Models(VLMs).
Its core inference engines include TurboMind Engine and PyTorch Engine. The former is developed by C++ and CUDA, striving for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

It supports LLMs and VLMs deployment on both Linux and Windows platform, with minimum requirement of CUDA version 11.3. Furthermore, it is compatible with the following NVIDIA GPUs:

- Volta(sm70): V100
- Turing(sm75): 20 series, T4
- Ampere(sm80,sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace(sm89): 40 series

## Install with pip (Recommend)

It is recommended installing lmdeploy using pip in a conda environment (python 3.9 - 3.13):

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
pip install lmdeploy
```

The default prebuilt package is compiled on **CUDA 12**. If CUDA 11+ (>=11.3) is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.10.2
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## Install nightly-build package with pip

The release frequency of LMDeploy is approximately once or twice monthly. If your desired feature has been merged to LMDeploy main branch but hasn't been published yet, you can experiment with the nightly-built package available [here](https://github.com/zhyncs/lmdeploy-build) according to your CUDA and Python versions

## Install from source

By default, LMDeploy will build with NVIDIA CUDA support, utilizing both the Turbomind and PyTorch backends. Before installing LMDeploy, ensure you have successfully installed the CUDA Toolkit.

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
pip install https://github.com/InternLM/lmdeploy/archive/refs/tags/v0.10.2.zip
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
