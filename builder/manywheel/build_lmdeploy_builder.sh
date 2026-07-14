#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)/builder
GPU_ARCH_VERSION=${GPU_ARCH_VERSION}
WITH_PUSH=${WITH_PUSH:-}

TARGET=cuda_final
DOCKER_TAG=manylinux_${MANY_LINUX_VERSION}-cuda${GPU_ARCH_VERSION}

DOCKER_IMAGE=openmmlab/lmdeploy-builder:${DOCKER_TAG}

# List of all build arguments (format: KEY=VALUE)
# Empty values will be automatically filtered out later
BUILD_ARGS=(
    "BASE_CUDA_VERSION=${GPU_ARCH_VERSION}"
    "HTTPS_PROXY=${HTTPS_PROXY:-}"
    "HTTP_PROXY=${HTTP_PROXY:-}"
    # Add more parameters here if needed
)

# Base Docker build command arguments
docker_build_args=(
    -t "${DOCKER_IMAGE}"
    --target "${TARGET}"
    -f "${TOPDIR}/manywheel/Dockerfile_${MANY_LINUX_VERSION}"
)

# Process build arguments: filter empty values and format as --build-arg
for arg in "${BUILD_ARGS[@]}"; do
    IFS='=' read -r key value <<< "$arg"  # Split KEY=VALUE
    if [[ -n "$value" ]]; then  # Only add non-empty values
        docker_build_args+=(--build-arg "$arg")
    fi
done

(
    set -x
    DOCKER_BUILDKIT=1 docker build "${docker_build_args[@]}" "${TOPDIR}"
)

if [[ "${WITH_PUSH}" == true ]]; then
    (
        set -x
        docker push "${DOCKER_IMAGE}"
    )
fi
