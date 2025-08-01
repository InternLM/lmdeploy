# Base images

ARG CUDA_VERSION=cu12

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cu12.8
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu124

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cu11
ENV CUDA_VERSION_SHORT=cu118

# Builder image
FROM ${CUDA_VERSION} AS builder
ARG PYTHON_VERSION=3.10
ARG NCCL_BRANCH=v2.26.6-1

# Should be in the lmdeploy root directory when building docker image
COPY . /opt/lmdeploy
WORKDIR /opt/lmdeploy

RUN --mount=type=cache,target=/root/.cache docker/build.sh
RUN --mount=type=cache,target=/root/.cache docker/prepare_wheel.sh

# Runtime image
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS cu12.8-base
ENV CUDA_VERSION_SHORT=cu128

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS cu12-base
ENV CUDA_VERSION_SHORT=cu124

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS cu11-base
ENV CUDA_VERSION_SHORT=cu118

FROM ${CUDA_VERSION}-base AS final
ARG PYTHON_VERSION=3.10

COPY --from=builder /wheels /wheels
COPY docker/install.sh /tmp/install.sh

RUN --mount=type=cache,target=/root/.cache /tmp/install.sh

RUN rm -rf /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl
COPY --from=builder /usr/local/include /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/include
COPY --from=builder /usr/local/nccl/lib /opt/py3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib
COPY --from=builder /usr/local/gdrcopy /usr/local/gdrcopy

# explicitly set ptxas path for triton
ENV PATH=/opt/py3/bin:$PATH
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV NCCL_LAUNCH_MODE=GROUP
