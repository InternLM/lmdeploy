#!/bin/bash -ex

apt-get update -y
apt-get install -y --no-install-recommends \
    software-properties-common wget curl ssh sudo \
    git-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-minimal python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv

pushd /opt >/dev/null
    python${PYTHON_VERSION} -m venv py3
popd >/dev/null

if [[ "${CUDA_VERSION_SHORT}" = "cu118" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-11-8
elif [[ "${CUDA_VERSION_SHORT}" = "cu124" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-12-4
elif [[ "${CUDA_VERSION_SHORT}" = "cu128" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-12-8
fi

apt-get clean -y
rm -rf /var/lib/apt/lists/*

export PATH=/opt/py3/bin:$PATH

pip install /wheels/*.whl --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}
