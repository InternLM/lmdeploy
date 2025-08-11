#!/bin/bash -ex

mkdir -p /wheels

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    pip install nvidia-nccl-cu12
else
    pip install nvidia-nccl-cu11
fi

python3 -m build -w -o /wheels -v .
