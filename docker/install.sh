#!/bin/bash -ex

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list
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

if [[ "${CUDA_VERSION_SHORT}" = "cu118" ]]; then
    FA_VERSION=2.7.3
    TORCH_VERSION="<2.7"
else
    FA_VERSION=2.8.3
    TORCH_VERSION=""
fi

pip install -U pip wheel setuptools

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    pip install nvidia-nvshmem-cu12
fi

pip install /wheels/*.whl torch${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}


if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]] && [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    pip install cuda-python dlblas
fi

PLATFORM="linux_x86_64"
PY_VERSION=$(python3 - <<'PY'
import torch, sys
torch_ver = '.'.join(torch.__version__.split('.')[:2])
cuda_ver  = torch.version.cuda.split('.')[0]
cxx11abi  = str(torch.compiled_with_cxx11_abi()).upper()
py_tag    = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"{torch_ver} {cuda_ver} {cxx11abi} {py_tag}")
PY
)

read TORCH_VER CUDA_VER CXX11ABI PY_TAG <<< "$PY_VERSION"

WHEEL="flash_attn-${FA_VERSION}+cu${CUDA_VER}torch${TORCH_VER}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-${PLATFORM}.whl"
BASE_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}"
FULL_URL="${BASE_URL}/${WHEEL}"

pip install "$FULL_URL"

# copy nccl
if [[ "${CUDA_VERSION_SHORT}" = "cu118" ]]; then
    rm -rf /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl
    cp -R /nccl /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/
fi
