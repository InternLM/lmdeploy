#!/bin/bash -ex

mkdir -p /wheels

if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    pip install nvidia-nccl-cu13
else
    pip install nvidia-nccl-cu12
fi
pip install --upgrade pip build
python3 -m build -w -o /wheels -v .
