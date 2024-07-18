# Installation

LMDeploy is a python library for compressing, deploying, and serving Large Language Models and Vision-Language Models.
Its core inference engines include TurboMind Engine and PyTorch Engine. The former is developed by C++ and CUDA, striving for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

It supports both Linux and Windows platform, with minimum requirement of CUDA version 11.3.

## Install with pip (Recommend)

You can install lmdeploy using pip (python 3.8 - 3.12) as follows:

```shell
pip install lmdeploy
```

The default prebuilt package is compiled on **CUDA 12**. If CUDA 11+ (>=11.3) is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.5.1
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## Install nightly-build package with pip

The release frequency of LMDeploy is approximately once or twice monthly. If your desired feature has been merged to LMDeploy main branch but hasn't been published yet, you can experiment with the nightly-built package available [here](https://github.com/zhyncs/lmdeploy-build) according to your CUDA and Python versions

## Install from the source

If you are using the PyTorch Engine for inference, the installation from the source is quite simple:

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
pip install -e .
```

But if you are using the TurboMind Engine, you have to build the source as shown below:

Clone LMDeploy source code and change to its root directory:

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

Run the following command to build the whl package according to your CUDA and Python versions.
Kindly select judiciously from the provided `docker_tag` options `{cuda12.1, cuda11.8}` and the Python version set `{py38, py39, py310, py311, py312}`.

```shell
docker_tag="cuda12.1"
py_version="py310"
output_dir="lmdeploy_wheel"
bash builder/manywheel/build_wheel.sh ${py_version} "manylinux2014_x86_64" ${docker_tag} ${output_dir}
```

After the whl is built successfully, you can install it by:

```shell
pip install builder/manywheel/${output_dir}/*.whl
```
