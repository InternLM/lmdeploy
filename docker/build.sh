#!/bin/bash -ex

mkdir -p /wheels /nccl

if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    pip install nvidia-nccl-cu13
elif [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    pip install nvidia-nccl-cu12
else
    NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"
    pushd /tmp >/dev/null
        git clone --depth=1 --branch ${NCCL_BRANCH} https://github.com/NVIDIA/nccl.git
        pushd nccl >/dev/null
            make NVCC_GENCODE="$NVCC_GENCODE" -j$(nproc) src.build
            mkdir -p /nccl/include /nccl/lib
            mv build/include/* /nccl/include/
            mv build/lib/lib* /nccl/lib/
        popd >/dev/null
    popd >/dev/null
    rm -rf /tmp/nccl
    export LD_LIBRARY_PATH=/nccl/lib:$LD_LIBRARY_PATH
fi

pip install --upgrade pip build
python3 -m build -w -o /wheels -v .
