#!/bin/bash -ex

mkdir -p /wheels

if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    pip install nvidia-nccl-cu13
else
    pip install nvidia-nccl-cu12
fi

python3 -m build -w -o /wheels -v .
