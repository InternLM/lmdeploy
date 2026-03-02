#!/bin/bash -ex

# install system packages
export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list
apt-get update -y
apt-get install -y --no-install-recommends \
    tzdata wget curl ssh sudo git-core vim libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1

if [[ ${PYTHON_VERSION} != "3.10" ]]; then
    apt-get install -y --no-install-recommends software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
fi

# install python, create virtual env
apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv

pushd /opt >/dev/null
    python${PYTHON_VERSION} -m venv py3
popd >/dev/null

# install CUDA build tools
if [[ "${CUDA_VERSION_SHORT}" = "cu126" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-12-6 numactl dkms
elif [[ "${CUDA_VERSION_SHORT}" = "cu128" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-12-8 numactl dkms
elif [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    apt-get install -y --no-install-recommends cuda-minimal-build-13-0 numactl dkms
fi

apt-get clean -y
rm -rf /var/lib/apt/lists/*

# install GDRCopy debs
if [ "$(ls -A /wheels/*.deb 2>/dev/null)" ]; then
    dpkg -i /wheels/*.deb
fi

# install python packages
export PATH=/opt/py3/bin:$PATH

pip install -U pip wheel setuptools

if [[ "${CUDA_VERSION_SHORT}" = "cu130" ]]; then
    pip install nvidia-nvshmem-cu13==3.4.5
else
    pip install nvidia-nvshmem-cu12==3.4.5
fi


pip install /wheels/*.whl
pip install cuda-python dlblas==0.0.7 dlslime==0.0.2.post1

# install pre-built flash attention wheel
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

if [[ "${CUDA_VER}" == "12" ]]; then
    FA_VERSION=2.8.3
    WHEEL="flash_attn-${FA_VERSION}+cu${CUDA_VER}torch${TORCH_VER}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-${PLATFORM}.whl"
    BASE_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}"
    FULL_URL="${BASE_URL}/${WHEEL}"

    pip install "$FULL_URL"
fi

# install pre-built flash attention 3 wheel
pip install ninja einops packaging
FA3_WHEELS_URL="https://windreamer.github.io/flash-attention3-wheels/${CUDA_VERSION_SHORT}_torch${TORCH_INFO}"
    pip install --no-index flash_attn_3 --find-links ${FA3_WHEELS_URL}


# install requirements/serve.txt dependencies such as timm
pip install -r /tmp/requirements/serve.txt

# copy nccl
if [[ "${CUDA_VERSION_SHORT}" = "cu118" ]]; then
    rm -rf /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl
    cp -R /nccl /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/
elif [[ "${CUDA_VERSION_SHORT}" = "cu128" ]]; then
    # As described in https://github.com/InternLM/lmdeploy/pull/4313,
    # window registration may cause memory leaks in NCCL 2.27, NCCL 2.28+ resolves the issue,
    # but turbomind engine will use nccl GIN for EP in future, which is brought in since 2.29
    pip install "nvidia-nccl-cu12>2.29"
fi
