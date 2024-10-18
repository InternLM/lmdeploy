# 安装

LMDeploy 是一个用于大型语言模型（LLMs）和视觉-语言模型（VLMs）压缩、部署和服务的 Python 库。
其核心推理引擎包括 TurboMind 引擎和 PyTorch 引擎。前者由 C++ 和 CUDA 开发，致力于推理性能的优化，而后者纯 Python 开发，旨在降低开发者的门槛。

LMDeploy 支持在 Linux 和 Windows 平台上部署 LLMs 和 VLMs，最低要求 CUDA 版本为 11.3。此外，它还与以下 NVIDIA GPU 兼容：

Volta(sm70): V100
Turing(sm75): 20 系列，T4
Ampere(sm80,sm86): 30 系列，A10, A16, A30, A100
Ada Lovelace(sm89): 40 系列

## 使用 pip 安装（推荐）

我们推荐在一个干净的conda环境下（python3.8 - 3.12），安装 lmdeploy：

```shell
conda create -n lmdeploy python=3.8 -y
conda activate lmdeploy
pip install lmdeploy
```

默认的预构建包是在 **CUDA 12** 上编译的。如果需要 CUDA 11+ (>=11.3)，你可以使用以下命令安装 lmdeploy：

```shell
export LMDEPLOY_VERSION=0.6.1
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## 使用 pip 安装夜间构建包

LMDeploy 的发布频率大约是每月一次或两次。如果你所需的功能已经被合并到 LMDeploy 的主分支但还没有发布，你可以环境中的 CUDA 和 Python 版本，尝试使用[这里](https://github.com/zhyncs/lmdeploy-build)提供的夜间构建包。

## 从源码安装

如果你使用 PyTorch 引擎进行推理，从源代码安装非常简单：

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
pip install -e .
```

但如果你使用 TurboMind 引擎，请参考以下说明编译源代码。我们强烈推荐使用 `openmmlab/lmdeploy:{tag}` docker 镜像作为编译安装的环境

**步骤 1** - 获取 LMDeploy 的 docker 镜像

```shell
docker pull openmmlab/lmdeploy:latest
```

```{note}
"openmmlab/lmdeploy:latest" 基于 "nvidia/cuda:12.4.1-devel-ubuntu22.04"。如果你在带有 cuda 11+ 驱动的平台上工作，请使用 "openmmlab/lmdeploy:latest-cu11"。
从 v0.5.3 开始，LMDeploy docker 镜像标签的模式是 "openmmlab/lmdeploy:{version}-cu(11|12)"。
```

**步骤 2** - 克隆 LMDeploy 源代码

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

**步骤 3** - 以交互模式启动 docker 容器

```shell
docker run --gpus all --net host --shm-size 16g -v $(pwd):/opt/lmdeploy --name lmdeploy -it openmmlab/lmdeploy:latest bin/bash
```

**步骤 4** - 编译与安装

```shell
cd /opt/lmdeploy
mkdir -p build && cd build
bash ../generate.sh make
make -j$(nproc) && make install
cd ..
pip install -e .
```
