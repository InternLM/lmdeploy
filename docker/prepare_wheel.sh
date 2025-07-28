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

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    GDRCOPY_VERSION=2.4.4
    NVSHMEM_VERSION=3.2.5-1
    DEEP_EP_VERSION=bdd119f
    FLASH_MLA_VERSION=9edee0c

    if [[ "${CUDA_VERSION_SHORT}" = "cu124" ]]; then
        DEEP_GEMM_VERSION=03d0be3
    else
        DEEP_GEMM_VERSION=1876566
    fi

    pushd /tmp >/dev/null
        curl -sSL "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz" | tar xz

        pushd gdrcopy-${GDRCOPY_VERSION} >/dev/null
            make prefix=/usr/local/gdrcopy -j $(nproc) install
        popd >/dev/null
        rm -rf gdrcopy-${GDRCOPY_VERSION}

        git clone https://github.com/deepseek-ai/DeepEP.git
        pushd DeepEP >/dev/null
            git checkout ${DEEP_EP_VERSION}
        popd >/dev/null


        # NVSHMEM
        curl -sSL "https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_${NVSHMEM_VERSION}.txz" | tar xz
        pushd nvshmem_src >/dev/null
            git apply /tmp/DeepEP/third-party/nvshmem.patch
            NVSHMEM_SHMEM_SUPPORT=0 \
            NVSHMEM_UCX_SUPPORT=0 \
            NVSHMEM_USE_NCCL=0 \
            NVSHMEM_MPI_SUPPORT=0 \
            NVSHMEM_IBGDA_SUPPORT=1 \
            NVSHMEM_PMIX_SUPPORT=0 \
            NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
            NVSHMEM_USE_GDRCOPY=1 \
            cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/usr/local/nvshmem -DMLX5_lib=/lib/x86_64-linux-gnu/libmlx5.so.1
            cmake --build build --target install --parallel $(nproc)
        popd >/dev/null
        rm -rf nvshmem_src

        NVSHMEM_DIR=/usr/local/nvshmem pip wheel -v --no-build-isolation --no-deps -w /wheels ./DeepEP
        rm -rf DeepEP

        pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}"
        pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}"

    popd >/dev/null
else
    mkdir -p /usr/local/gdrcopy
    mkdir -p /usr/local/nvshmem
fi
