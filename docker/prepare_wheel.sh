#!/bin/bash -ex

export PATH=/opt/py3/bin:$PATH

if [[ "${CUDA_VERSION_SHORT}" = "cu118" ]]; then
    TORCH_VERSION="<2.7"
else
    TORCH_VERSION=""
fi

pip install "cmake<4.0" wheel ninja setuptools packaging
pip install torch${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}

if [[ ${PYTHON_VERSION} = "3.13" ]]; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    . "$HOME/.cargo/env"

    pip install setuptools_rust
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/google/sentencepiece.git@v0.2.0#subdirectory=python"
fi

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then

    if [[ "${CUDA_VERSION_SHORT}" = "cu124" ]]; then
        DEEP_GEMM_VERSION=03d0be3
        FLASH_MLA_VERSION=9edee0c
    else
        DEEP_GEMM_VERSION=79f48ee
        FLASH_MLA_VERSION=c759027
    fi

    DEEP_EP_VERSION=26cf250
    pip install nvidia-nvshmem-cu12

    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_VERSION}"
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}"
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}"
fi
