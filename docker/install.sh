#!/bin/bash -ex

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
apt-get update -y
apt-get install -y --no-install-recommends \
    tzdata wget curl ssh sudo git-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1

if [[ ${PYTHON_VERSION} != "3.10" ]]; then
    apt-get install -y --no-install-recommends software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
fi

apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv

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

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]] && [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    pip install -U pip wheel setuptools
    pip install cuda-python dlblas
fi
