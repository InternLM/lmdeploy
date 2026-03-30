# Base images
ARG IMAGE_TYPE=final
ARG CUDA_VERSION=cu12

FROM nvidia/cuda:13.0.2-devel-ubuntu22.04 AS cu13
ENV CUDA_VERSION_SHORT=cu130

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cu12.8
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu126

# Builder image
FROM ${CUDA_VERSION} AS dev
ARG PYTHON_VERSION=3.12

ENV PATH=/opt/py3/bin:/root/.local/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN --mount=type=cache,target=/root/.cache \
    sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tzdata wget curl openssh-server ssh sudo git-core \
        libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1 \
        libssl-dev pkg-config vim rapidjson-dev libgoogle-glog-dev gdb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    wget -qO- https://astral.sh/uv/install.sh | sh && \
    uv venv -p python${PYTHON_VERSION} --seed /opt/py3 && \
    pip install --upgrade pip build

# Pip deps only (cache until requirements/runtime_cuda.txt changes). No full repo COPY yet.
FROM dev AS runtime-deps-pip
ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}

COPY requirements/runtime_cuda.txt /tmp/runtime_cuda.txt

RUN --mount=type=cache,target=/root/.cache \
    pip install -r /tmp/runtime_cuda.txt --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}

RUN mkdir -p /wheels

# Does not use lmdeploy sources; base on runtime-deps-pip so it is not invalidated by app code changes.
FROM runtime-deps-pip AS thirdparty-wheels

COPY docker/prepare_3rdparty_wheel.sh /tmp/prepare_3rdparty_wheel.sh

RUN --mount=type=cache,target=/root/.cache \
    bash /tmp/prepare_3rdparty_wheel.sh

# Full tree for local package build
FROM runtime-deps-pip AS lmdeploy-wheel

COPY . /opt/lmdeploy
WORKDIR /opt/lmdeploy

RUN --mount=type=cache,target=/root/.cache \
    docker/build.sh

FROM runtime-deps-pip AS wheels

COPY --from=thirdparty-wheels /wheels /wheels
COPY --from=lmdeploy-wheel /wheels /wheels

# Runtime image
FROM nvidia/cuda:13.0.2-base-ubuntu22.04 AS cu13-base
ENV CUDA_VERSION_SHORT=cu130

FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS cu12.8-base
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS cu12-base
ENV CUDA_VERSION_SHORT=cu126

FROM ${CUDA_VERSION}-base AS final
ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Some dependencies such as timm (required by InternVL models) are listed in requirements/serve.txt.
COPY requirements/serve.txt /tmp/requirements/serve.txt
COPY docker/install.sh /tmp/install.sh
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/wheels,from=wheels,source=/wheels \
    /tmp/install.sh

# explicitly set ptxas path for triton
ENV PATH=/opt/py3/bin:$PATH
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
FROM ${IMAGE_TYPE}
