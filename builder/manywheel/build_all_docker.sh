#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)/builder

for cuda_version in 11.8; do
    MANY_LINUX_VERSION=2014 GPU_ARCH_VERSION="${cuda_version}" "${TOPDIR}/manywheel/build_docker.sh"
done
