#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)/builder
GPU_ARCH_VERSION=${GPU_ARCH_VERSION}
WITH_PUSH=${WITH_PUSH:-}

TARGET=cuda_final
DOCKER_TAG=cuda${GPU_ARCH_VERSION}
DOCKER_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=9"
DOCKER_TAG=cuda${GPU_ARCH_VERSION}

DOCKER_IMAGE=openmmlab/lmdeploy-builder:${DOCKER_TAG}
if [[ -n ${MANY_LINUX_VERSION} ]]; then
    DOCKERFILE_SUFFIX=_${MANY_LINUX_VERSION}
else
    DOCKERFILE_SUFFIX=''
fi

(
    set -x
    DOCKER_BUILDKIT=1 docker build \
        -t "${DOCKER_IMAGE}" \
        ${DOCKER_BUILD_ARG} \
        --target "${TARGET}" \
        -f "${TOPDIR}/manywheel/Dockerfile${DOCKERFILE_SUFFIX}" \
        "${TOPDIR}"
)

if [[ "${WITH_PUSH}" == true ]]; then
    (
        set -x
        docker push "${DOCKER_IMAGE}"
    )
fi
