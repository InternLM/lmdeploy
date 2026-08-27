#!/bin/bash -ex

# Skip system setup if virtual env already exists (e.g., in dev image)
if [ ! -f "/opt/py3/bin/python" ]; then
    # install system packages
    export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
    if [ -f /etc/apt/sources.list ]; then
        sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list
    fi
    if [ -d /etc/apt/sources.list.d ]; then
        find /etc/apt/sources.list.d -type f -name '*.sources' -exec \
            sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' {} +
    fi
    apt-get update -y
    apt-get install -y --no-install-recommends \
        tzdata wget curl ssh sudo git-core vim libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1

    # Ubuntu may expose the interpreter package without the matching -dev and
    # -venv packages. Check the complete set before deciding whether the PPA
    # is needed.
    python_packages=(
        "python${PYTHON_VERSION}"
        "python${PYTHON_VERSION}-dev"
        "python${PYTHON_VERSION}-venv"
    )
    need_python_ppa=0
    for package in "${python_packages[@]}"; do
        if ! apt-cache show "${package}" >/dev/null 2>&1; then
            need_python_ppa=1
            break
        fi
    done
    if ((need_python_ppa)); then
        apt-get install -y --no-install-recommends software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update -y
    fi

    # install python, create virtual env
    apt-get install -y --no-install-recommends "${python_packages[@]}"

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
fi

# install GDRCopy debs
if [ "$(ls -A /wheels/*.deb 2>/dev/null)" ]; then
    dpkg -i /wheels/*.deb
fi

# install python packages
export PATH=/opt/py3/bin:$PATH

pip install -U pip wheel setuptools

# In the production image, install torch/torchvision from the matching CUDA
# index first. Installing all local wheels directly would otherwise let pip
# resolve torch from the default PyPI index.
if [ -f /tmp/requirements/runtime_cuda.txt ]; then
    grep -E '^torch(vision)?([<>=]|$)' \
        /tmp/requirements/runtime_cuda.txt > /tmp/requirements/torch.txt
    pip install -r /tmp/requirements/torch.txt \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION}"
    pip install -r /tmp/requirements/runtime_cuda.txt
fi

if [[ "${CUDA_VERSION_SHORT}" == cu13* ]]; then
    pip install nvidia-nvshmem-cu13==3.4.5
else
    pip install nvidia-nvshmem-cu12==3.4.5
fi

for package in deep_ep deep_gemm flash_attn_3; do
    if ! compgen -G "/wheels/${package}-*.whl" >/dev/null; then
        echo "${package} wheel is missing from /wheels" >&2
        exit 1
    fi
done
pip install /wheels/*.whl
python - <<'PY'
import deep_ep
import deep_gemm
import lmdeploy.pytorch.third_party.flash_attn_interface
PY
pip install dlslime==0.0.2.post1

pip install ninja einops packaging

# install requirements/serve.txt dependencies such as timm
if [ -f /tmp/requirements/serve.txt ]; then
    pip install -r /tmp/requirements/serve.txt
fi

# PyTorch cu130 pins an older NCCL release in its wheel metadata. Check the
# declared dependencies before intentionally upgrading NCCL for GIN support.
pip check

if [[ "${CUDA_VERSION_SHORT}" = "cu128" ]]; then
    pip install "nvidia-nccl-cu12>=2.30.4"
elif [[ "${CUDA_VERSION_SHORT}" == cu13* ]]; then
    # tile-kernels (for DeepSeek-V4) requires a cu13 tilelang build.
    pip install "tile-kernels==1.0.0"
    pip install "nvidia-nccl-cu13>=2.30.4"
fi

python - <<'PY'
import os
from importlib.metadata import distributions, version

import torch
from packaging.version import Version

expected = os.environ['PYTORCH_CUDA_VERSION']
actual = torch.version.cuda
installed = {
    dist.metadata['Name'].lower().replace('_', '-')
    for dist in distributions()
    if dist.metadata['Name']
}
if expected == 'cu126':
    assert actual is not None and actual.startswith('12.6'), actual
    nccl_package = 'nvidia-nccl-cu12'
    nccl_minimum = Version('2.29')
    unexpected = sorted(name for name in installed if name.endswith('-cu13'))
elif expected == 'cu130':
    assert actual is not None and actual.startswith('13.0'), actual
    nccl_package = 'nvidia-nccl-cu13'
    nccl_minimum = Version('2.30.4')
    unexpected = sorted(name for name in installed if name.endswith('-cu12'))
else:
    raise AssertionError(f'unsupported PYTORCH_CUDA_VERSION: {expected}')
assert not unexpected, f'unexpected CUDA packages: {unexpected}'
nccl_version = version(nccl_package)
if expected == 'cu126':
    assert Version(nccl_version) > nccl_minimum, nccl_version
else:
    assert Version(nccl_version) >= nccl_minimum, nccl_version
print(f'Validated torch {torch.__version__} with CUDA {actual} and '
      f'{nccl_package} {nccl_version}')
PY
