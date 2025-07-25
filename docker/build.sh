#!/bin/bash -ex

apt-get update -y
apt-get install -y --no-install-recommends \
    software-properties-common wget curl openssh-server ssh sudo \
    git-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION}-minimal python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
apt-get clean -y
rm -rf /var/lib/apt/lists/*

pushd /opt >/dev/null
    python${PYTHON_VERSION} -m venv py3
popd >/dev/null

export PATH=/opt/py3/bin:$PATH

NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"

pushd /tmp >/dev/null
    git clone --depth=1 --branch ${NCCL_BRANCH} https://github.com/NVIDIA/nccl.git
    pushd nccl >/dev/null
        make NVCC_GENCODE="$NVCC_GENCODE" -j$(nproc) src.build
        mv build/include/* /usr/local/include
        mkdir -p /usr/local/nccl/lib
        mv build/lib/lib* /usr/local/nccl/lib/
    popd >/dev/null
popd >/dev/null
rm -rf /tmp/nccl

export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH

pip install --upgrade pip build
python3 -m build -w -o /wheels -v .
