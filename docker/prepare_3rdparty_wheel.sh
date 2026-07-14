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
DEEP_GEMM_VERSION=c9f8b34  # v2.1.1.post3
FLASH_MLA_VERSION=1408756  # no release, pick the latest commit

# DeepEP
if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl":${CPLUS_INCLUDE_PATH}
    pip install nvidia-nvshmem-cu13==3.4.5
else
    pip install nvidia-nvshmem-cu12==3.4.5
fi
pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_VERSION}"

# DeepGEMM
pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}"

# FlashMLA
# sm100 compilation for Flash MLA requires NVCC 12.9 or higher
FLASH_MLA_DISABLE_SM100=1 pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}"

# flash_attn_3 (prebuilt wheels; CUDA + torch must match this image)
TORCH_VER=$(python3 -c "import torch; print(''.join(torch.__version__.split('+')[0].split('.')))")
FA3_WHEELS_URL="https://windreamer.github.io/flash-attention3-wheels/${CUDA_VERSION_SHORT}_torch${TORCH_VER}"
pip download --no-deps -d /wheels "flash_attn_3" --find-links "${FA3_WHEELS_URL}"

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
