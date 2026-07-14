#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)/builder
MANY_LINUX_VERSION=${MANY_LINUX_VERSION:-2_28}
CUDA_VERSION=${CUDA_VERSION:-12.8}

for cuda_version in ${CUDA_VERSION}; do
    MANY_LINUX_VERSION="${MANY_LINUX_VERSION}" \
        GPU_ARCH_VERSION="${cuda_version}" \
        "${TOPDIR}/manywheel/build_lmdeploy_builder.sh"
done
