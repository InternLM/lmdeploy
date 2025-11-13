# Base images
ARG IMAGE_TYPE=final
ARG CUDA_VERSION=cu12

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cu12.8
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu124

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cu11
ENV CUDA_VERSION_SHORT=cu118

# Builder image
FROM ${CUDA_VERSION} AS dev
ARG PYTHON_VERSION=3.10
ARG NCCL_BRANCH=v2.26.6-1

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

FROM dev AS builder

# Should be in the lmdeploy root directory when building docker image
COPY . /opt/lmdeploy
WORKDIR /opt/lmdeploy

RUN --mount=type=cache,target=/root/.cache \
    docker/build.sh && \
    docker/prepare_wheel.sh

# Runtime image
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS cu12.8-base
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS cu12-base
ENV CUDA_VERSION_SHORT=cu124

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS cu11-base
ENV CUDA_VERSION_SHORT=cu118

FROM ${CUDA_VERSION}-base AS final
ARG PYTHON_VERSION=3.10

COPY docker/install.sh /tmp/install.sh

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/wheels,from=builder,source=/wheels \
    --mount=type=cache,target=/nccl,from=builder,source=/nccl \
    /tmp/install.sh

# explicitly set ptxas path for triton
ENV PATH=/opt/py3/bin:$PATH
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV LD_LIBRARY_PATH=/nccl/lib:$LD_LIBRARY_PATH
FROM ${IMAGE_TYPE}
