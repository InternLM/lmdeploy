#!/usr/bin/env bash
set -eux

PYTHON_VERSION="$1"
PLAT_NAME="$2"
DOCKER_TAG="$3"
OUTPUT_DIR="$4"

DOCKER_IMAGE="openmmlab/lmdeploy-builder:${DOCKER_TAG}"

cd "$(dirname "$0")"  # move inside the script directory
mkdir -p "${OUTPUT_DIR}"
# docker pull ${docker_image}
docker run --rm -it \
    --env PYTHON_VERSION="${PYTHON_VERSION}" \
    --env PLAT_NAME="${PLAT_NAME}" \
    --volume "$(pwd)/${OUTPUT_DIR}:/lmdeploy_build" \
    --volume "$(pwd)/entrypoint_build.sh:/entrypoint_build.sh" \
    --entrypoint /entrypoint_build.sh \
    ${DOCKER_IMAGE}
