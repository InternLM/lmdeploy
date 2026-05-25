#!/bin/bash -ex

mkdir -p /wheels

if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    pip install nvidia-nccl-cu13==2.30.3
else
    pip install nvidia-nccl-cu12==2.30.3
fi

# Hint FindNCCL.cmake to use NCCL from the host python env; pip's isolated
# build env does not see /opt/py3's site-packages.
NCCL_DIR=$(python3 -c "import importlib.util; print(importlib.util.find_spec('nvidia.nccl').submodule_search_locations[0])")
export NCCL_INCLUDE_DIR="${NCCL_DIR}/include"
export NCCL_LIB_DIR="${NCCL_DIR}/lib"
export NCCL_VERSION=$(ls "${NCCL_LIB_DIR}"/libnccl.so.* | sed -E 's/.*libnccl\.so\.([0-9]+)$/\1/' | sort -n | tail -1)

python3 -m build -w -o /wheels -v .
