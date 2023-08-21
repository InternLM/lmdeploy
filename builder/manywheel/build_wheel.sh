#!/usr/bin/env bash
set -eux

PYTHON_VERSION="$1"
PLAT_NAME="$2"
DOCKER_TAG="$3"
OUTPUT_DIR="$4"

GIT_REMOTE=${GIT_REMOTE:-https://github.com/InternLM/lmdeploy}
GIT_BRANCH=${GIT_BRANCH:-main}
DOCKER_IMAGE="openmmlab/lmdeploy-builder:${DOCKER_TAG}"
export USERID=$(id -u)
export GROUPID=$(id -g)

cd "$(dirname "$0")"  # move inside the script directory
mkdir -p "${OUTPUT_DIR}"
docker pull ${DOCKER_IMAGE}
docker run --rm -it \
    --env PYTHON_VERSION="${PYTHON_VERSION}" \
    --env PLAT_NAME="${PLAT_NAME}" \
    --env USERID="${USERID}" \
    --env GROUPID="${GROUPID}" \
    --env GIT_BRANCH="${GIT_BRANCH}" \
    --env GIT_REMOTE="${GIT_REMOTE}" \
    --volume "$(pwd)/${OUTPUT_DIR}:/lmdeploy_build" \
    --volume "$(pwd)/entrypoint_build.sh:/entrypoint_build.sh" \
    --entrypoint /entrypoint_build.sh \
    ${DOCKER_IMAGE}
