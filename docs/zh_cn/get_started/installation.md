# 安装

LMDeploy 是一个用于大型语言模型（LLMs）和视觉-语言模型（VLMs）压缩、部署和服务的 Python 库。
其核心推理引擎包括 TurboMind 引擎和 PyTorch 引擎。前者由 C++ 和 CUDA 开发，致力于推理性能的优化，而后者纯 Python 开发，旨在降低开发者的门槛。

LMDeploy 支持在 Linux 和 Windows 平台上部署 LLMs 和 VLMs，最低要求 CUDA 版本为 11.3。此外，它还与以下 NVIDIA GPU 兼容：

Volta(sm70): V100
Turing(sm75): 20 系列，T4
Ampere(sm80,sm86): 30 系列，A10, A16, A30, A100
Ada Lovelace(sm89): 40 系列

## 使用 pip 安装（推荐）

我们推荐在一个干净的conda环境下（python3.9 - 3.13），安装 lmdeploy：

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
pip install lmdeploy
```

默认的预构建包是在 **CUDA 12** 上编译的。如果需要 CUDA 11+ (>=11.3)，你可以使用以下命令安装 lmdeploy：

```shell
export LMDEPLOY_VERSION=0.10.2
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## 使用 pip 安装夜间构建包

LMDeploy 的发布频率大约是每月一次或两次。如果你所需的功能已经被合并到 LMDeploy 的主分支但还没有发布，你可以环境中的 CUDA 和 Python 版本，尝试使用[这里](https://github.com/zhyncs/lmdeploy-build)提供的夜间构建包。

## 从源码安装

默认情况下，LMDeploy 将面向 NVIDIA CUDA 环境进行编译安装，并同时启用 Turbomind 和 PyTorch 两种后端引擎。在安装 LMDeploy 之前，请确保已成功安装 CUDA 工具包。

成功安装 CUDA 工具包后，您可以使用以下单行命令构建并安装 LMDeploy：

```shell
pip install git+https://github.com/InternLM/lmdeploy.git
```

您还可以通过设置 `DISABLE_TURBOMIND` 环境变量，显式禁用 Turbomind 后端，以避免 CUDA 编译：

```shell
DISABLE_TURBOMIND=1 pip install git+https://github.com/InternLM/lmdeploy.git
```

如果您希望使用特定版本，而不是 LMDeploy 的 `main` 分支，可以在命令行中指定：

```shell
pip install https://github.com/InternLM/lmdeploy/archive/refs/tags/v0.10.2.zip
```

如果您希望构建支持昇腾、寒武纪或沐熙的 LMDeploy，请使用相应的 `LMDEPLOY_TARGET_DEVICE` 环境变量进行安装。

LMDeploy 也支持在 AMD GPU 的 ROCm 环境中安装。

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
