# Installation

LMDeploy is a python library for compressing, deploying, and serving Large Language Models(LLMs) and Vision-Language Models(VLMs).
Its core inference engines include TurboMind Engine and PyTorch Engine. The former is developed by C++ and CUDA, striving for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

It supports LLMs and VLMs deployment on both Linux and Windows platform, with minimum requirement of CUDA version 11.3. Furthermore, it is compatible with the following NVIDIA GPUs:

- Volta(sm70): V100
- Turing(sm75): 20 series, T4
- Ampere(sm80,sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace(sm89): 40 series

## Install with pip (Recommend)

It is recommended installing lmdeploy using pip in a conda environment (python 3.8 - 3.12):

```shell
conda create -n lmdeploy python=3.8 -y
conda activate lmdeploy
pip install lmdeploy
```

The default prebuilt package is compiled on **CUDA 12**. If CUDA 11+ (>=11.3) is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.6.1
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## Install nightly-build package with pip

The release frequency of LMDeploy is approximately once or twice monthly. If your desired feature has been merged to LMDeploy main branch but hasn't been published yet, you can experiment with the nightly-built package available [here](https://github.com/zhyncs/lmdeploy-build) according to your CUDA and Python versions

## Install from source

If you are using the PyTorch Engine for inference, the installation from the source is quite simple:

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
pip install -e .
```

But if you are using the TurboMind Engine, you have to build the source as shown below. The `openmmlab/lmdeploy:{tag}` docker image is strongly recommended.

**Step 1** - Get the docker image of LMDeploy

```shell
docker pull openmmlab/lmdeploy:latest
```

```{note}
The "openmmlab/lmdeploy:latest" is based on "nvidia/cuda:12.4.1-devel-ubuntu22.04". If you are working on a platform with cuda 11+ driver, please use "openmmlab/lmdeploy:latest-cu11".
The pattern of the LMDeploy docker image tag is "openmmlab/lmdeploy:{version}-cu(11|12)" since v0.5.3.
```

**Step 2** - Clone LMDeploy source code and change to its root directory

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

**Step 3** - launch docker container in interactive mode

```shell
docker run --gpus all --net host --shm-size 16g -v $(pwd):/opt/lmdeploy --name lmdeploy -it openmmlab/lmdeploy:latest bin/bash
```

**Step 4** - build and installation

```shell
cd /opt/lmdeploy
mkdir -p build && cd build
bash ../generate.sh make
make -j$(nproc) && make install
cd ..
pip install -e .
```
