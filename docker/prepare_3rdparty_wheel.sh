#!/bin/bash -ex

export PATH=/opt/py3/bin:$PATH

pip install "cmake<4.0" wheel ninja setuptools packaging

if [[ ${PYTHON_VERSION} = "3.13" ]]; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    . "$HOME/.cargo/env"

    pip install setuptools_rust
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/google/sentencepiece.git@v0.2.0#subdirectory=python"
fi

GDRCOPY_VERSION=2.5.1
DEEP_EP_VERSION=9af0e0d  # v1.2.1
DEEP_GEMM_VERSION=88965b0
FLASH_MLA_VERSION=9241ae3
FLASH_KDA_VERSION=d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72
FAST_HADAMARD_TRANSFORM_VERSION=v1.1.0.post2

# DeepEP
if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl":${CPLUS_INCLUDE_PATH}
    pip install nvidia-nvshmem-cu13==3.4.5
    pip install mooncake-transfer-engine-cuda13==0.3.12.post1
else
    pip install nvidia-nvshmem-cu12==3.4.5
    pip install mooncake-transfer-engine==0.3.12.post1
fi
pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_VERSION}"

# DeepGEMM
pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}"

# FlashMLA
# sm100 compilation for Flash MLA requires NVCC 12.9 or higher
FLASH_MLA_DISABLE_SM100=1 pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}"

# FlashKDA requires CUDA >= 12.9; the CUDA 12 image uses Toolkit 12.8.
# Build explicitly for supported GPUs because Docker builders have no GPU.
if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    FLASH_KDA_CUDA_ARCHS=all NVCC_THREADS=2 MAX_JOBS=2 \
        pip wheel -v --no-build-isolation --no-deps -w /wheels \
        "git+https://github.com/MoonshotAI/FlashKDA.git@${FLASH_KDA_VERSION}"
fi

# fast_hadamard_transform
pip wheel -v --no-build-isolation --no-deps -w /wheels \
    "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@${FAST_HADAMARD_TRANSFORM_VERSION}"

# flash_attn_3 (official, LibTorch ABI-stable wheels for torch >= 2.9).
# Select the FA3 channel matching the PyTorch binary. CUDA 13 FA3 wheels are
# published in the cu130 channel.
FA3_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-${CUDA_VERSION_SHORT}}"
if [[ "${FA3_CUDA_VERSION}" == cu13* ]]; then
    FA3_CUDA_VERSION=cu130
fi
pip download --no-deps --only-binary=:all: -d /wheels \
    "flash-attn-3==3.0.0" \
    --index-url "https://download.pytorch.org/whl/${FA3_CUDA_VERSION}"

# Resolve the latest stable release via its redirect, avoiding the GitHub API's
# unauthenticated rate limit on shared Docker/CI builders.
ROUTER_RELEASE_URL=$(curl -fsSL --retry 3 -o /dev/null -w '%{url_effective}' \
    https://github.com/caikun-pjlab/router/releases/latest) || exit 1
if [[ "${ROUTER_RELEASE_URL}" != https://github.com/caikun-pjlab/router/releases/tag/* ]]; then
    echo "Unexpected router release URL: ${ROUTER_RELEASE_URL}" >&2
    exit 1
fi
ROUTER_TAG=${ROUTER_RELEASE_URL##*/}
ROUTER_VERSION=${ROUTER_TAG#v}
ROUTER_WHEEL="lmdeploy_router-${ROUTER_VERSION}-py3-none-linux_x86_64.whl"
wget -q -O "/wheels/${ROUTER_WHEEL}" \
    "https://github.com/caikun-pjlab/router/releases/download/${ROUTER_TAG}/${ROUTER_WHEEL}" || exit 1

# GDRCopy debs
apt-get update -y \
&& apt-get install -y --no-install-recommends build-essential devscripts debhelper fakeroot pkg-config dkms

wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
&& tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
&& cd gdrcopy-${GDRCOPY_VERSION}/packages \
&& CUDA=/usr/local/cuda ./build-deb-packages.sh \
&& mv ./*.deb /wheels

# Clean up build artifacts
cd / && rm -rf gdrcopy-${GDRCOPY_VERSION}
apt-get clean -y && rm -rf /var/lib/apt/lists/*
