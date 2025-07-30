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
    pip wheel -v --no-build-isolation --no-deps -w /wheels --use-deprecated=legacy-resolver outlines_core==0.1.26
fi

mkdir -p /usr/local/gdrcopy
if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    FLASH_MLA_VERSION=9edee0c

    if [[ "${CUDA_VERSION_SHORT}" = "cu124" ]]; then
        DEEP_GEMM_VERSION=03d0be3
    else
        DEEP_GEMM_VERSION=1876566
    fi

    if [[ ${PYTHON_VERSION} != "3.9" ]]; then
        GDRCOPY_VERSION=2.4.4
        DEEP_EP_VERSION=bdd119f
        pip install nvidia-nvshmem-cu12

        pushd /tmp >/dev/null
            curl -sSL "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz" | tar xz
            pushd gdrcopy-${GDRCOPY_VERSION} >/dev/null
                make prefix=/usr/local/gdrcopy -j $(nproc) install
            popd >/dev/null
            rm -rf gdrcopy-${GDRCOPY_VERSION}
        popd >/dev/null

        pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_VERSION}"
    fi
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}"
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}"
fi
