#!/bin/bash -ex

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
apt-get update -y
apt-get install -y --no-install-recommends \
    tzdata wget curl openssh-server ssh sudo git-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1 libssl-dev pkg-config

if [[ ${PYTHON_VERSION} != "3.10" ]]; then
    apt-get install -y --no-install-recommends software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
fi

apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
apt-get clean -y
rm -rf /var/lib/apt/lists/*

pushd /opt >/dev/null
    python${PYTHON_VERSION} -m venv py3
popd >/dev/null

export PATH=/opt/py3/bin:$PATH
mkdir -p /wheels

pip install --upgrade pip build
if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    pip install nvidia-nccl-cu12
else
    pip install nvidia-nccl-cu11
fi

python3 -m build -w -o /wheels -v .
