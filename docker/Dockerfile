# Base images
ARG IMAGE_TYPE=final
ARG CUDA_VERSION=cu128
ARG BUILD_ROUTER=1

FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS cu130
ENV CUDA_VERSION_SHORT=cu130
ENV PYTORCH_CUDA_VERSION=cu130

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS cu128
ENV CUDA_VERSION_SHORT=cu128
# PyTorch 2.12 no longer publishes cu128 wheels; cu126 is its CUDA 12 build.
ENV PYTORCH_CUDA_VERSION=cu126

# Backward-compatible stage aliases.
FROM cu130 AS cu13
FROM cu128 AS cu12.8
FROM cu128 AS cu12

# Builder image
FROM ${CUDA_VERSION} AS dev
ARG PYTHON_VERSION=3.12
ARG BUILD_ROUTER=1

ENV PATH=/opt/py3/bin:/root/.local/bin:/root/.cargo/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN --mount=type=cache,target=/root/.cache \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list; \
    fi && \
    if [ -d /etc/apt/sources.list.d ]; then \
        find /etc/apt/sources.list.d -type f -name '*.sources' -exec \
            sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' {} +; \
    fi && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tzdata wget curl openssh-server ssh sudo git-core \
        libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1 \
        libssl-dev pkg-config vim rapidjson-dev libgoogle-glog-dev gdb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    wget -qO- https://astral.sh/uv/install.sh | sh && \
    uv venv -p python${PYTHON_VERSION} --seed /opt/py3 && \
    pip install --upgrade pip build && \
    if [ "${BUILD_ROUTER}" != "0" ]; then \
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; \
    fi

# Pip deps only (cache until requirements/common.txt or runtime_cuda.txt changes). No full repo COPY yet.
FROM dev AS runtime-deps-pip
ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}

COPY requirements/common.txt requirements/runtime_cuda.txt /tmp/requirements/

RUN --mount=type=cache,target=/root/.cache \
    grep -E '^torch(vision)?([<>=]|$)' /tmp/requirements/runtime_cuda.txt > /tmp/requirements/torch.txt && \
    pip install -r /tmp/requirements/torch.txt --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION} && \
    pip install -r /tmp/requirements/runtime_cuda.txt

RUN mkdir -p /wheels

# Does not use lmdeploy sources; base on runtime-deps-pip so it is not invalidated by app code changes.
FROM runtime-deps-pip AS thirdparty-wheels

COPY docker/prepare_3rdparty_wheel.sh /tmp/prepare_3rdparty_wheel.sh

RUN --mount=type=cache,target=/root/.cache \
    bash /tmp/prepare_3rdparty_wheel.sh

# Full tree for local package build.
# Base on thirdparty-wheels (not runtime-deps-pip) to force buildx to build
# this stage only after thirdparty-wheels finishes. Both stages run heavy nvcc
# compilation; building them in parallel OOMs the 16GB CI runner (exit 143 /
# runner shutdown). Serializing halves peak memory. Only /wheels is consumed
# downstream, so the final image is unchanged.
FROM thirdparty-wheels AS lmdeploy-wheel

COPY . /opt/lmdeploy
WORKDIR /opt/lmdeploy

RUN --mount=type=cache,target=/root/.cache \
    docker/build.sh

# Build the standalone Rust router in the same image by default. Pass
# --build-arg BUILD_ROUTER=0 to skip Rust installation and compilation. The
# disabled branch posts a small marker because later bind mounts require the
# same artifact path in both variants.
FROM dev AS router-builder
ARG BUILD_ROUTER=1
ENV PATH=/opt/py3/bin:/root/.local/bin:/root/.cargo/bin:${PATH}
WORKDIR /src
COPY lmdeploy/router/Cargo.toml lmdeploy/router/Cargo.lock ./
COPY lmdeploy/router/src ./src
RUN if [ "${BUILD_ROUTER}" != "0" ]; then \
        cargo build --release --locked --no-default-features --bin lmdeploy-router && \
        install -D -m 0755 target/release/lmdeploy-router /src/router/lmdeploy-router; \
    else \
        mkdir -p /src/router && printf 'router disabled\n' > /src/router/lmdeploy-router; \
    fi

FROM runtime-deps-pip AS wheels

COPY --from=thirdparty-wheels /wheels /wheels
COPY --from=lmdeploy-wheel /wheels /wheels
# Keep a stable artifact path across enabled and disabled builds.
RUN --mount=type=bind,target=/router-build-stage,from=router-builder,source=/src/router \
    cp /router-build-stage/lmdeploy-router /wheels/lmdeploy-router

# Runtime image
FROM nvidia/cuda:13.0.3-base-ubuntu24.04 AS cu130-base
ENV CUDA_VERSION_SHORT=cu130
ENV PYTORCH_CUDA_VERSION=cu130

FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS cu128-base
ENV CUDA_VERSION_SHORT=cu128
ENV PYTORCH_CUDA_VERSION=cu126

# Backward-compatible stage aliases.
FROM cu130-base AS cu13-base
FROM cu128-base AS cu12.8-base
FROM cu128-base AS cu12-base

FROM ${CUDA_VERSION}-base AS final
ARG PYTHON_VERSION=3.12
ARG BUILD_ROUTER=1
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Install the CUDA runtime requirements explicitly so torch is selected from
# the matching PyTorch CUDA index before local wheels resolve dependencies.
# Some dependencies such as timm (required by InternVL models) are listed in requirements/serve.txt.
COPY requirements/common.txt requirements/runtime_cuda.txt requirements/serve.txt /tmp/requirements/
COPY docker/install.sh /tmp/install.sh
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,target=/wheels,from=wheels,source=/wheels \
    /tmp/install.sh

# explicitly set ptxas path for triton
ENV PATH=/opt/py3/bin:$PATH
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
RUN --mount=type=bind,target=/wheels-build-stage,from=wheels,source=/wheels \
    if [ "${BUILD_ROUTER}" != "0" ]; then \
        install -m 0755 /wheels-build-stage/lmdeploy-router /opt/lmdeploy-router; \
    fi
ENV PATH=/opt:${PATH}
FROM ${IMAGE_TYPE}
