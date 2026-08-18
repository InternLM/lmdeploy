---
name: docker-build
description: Build a CUDA 13.0 or CUDA 12.8 LMDeploy Docker image and push it to
  the inner registry.
disable-model-invocation: true
---

# Docker Build & Push

Build an LMDeploy Docker image and push it to the inner registry.

## Prerequisites

Before starting, verify all three environment variables are set:

```bash
echo $LMDEPLOY_REGISTRY    # inner registry server address
echo $REGISTRY_USER        # registry login username
test -n "$REGISTRY_PASSWORD" && echo "<set>" || echo "<missing>"  # registry login password
```

If any are missing, stop and tell the user to set them before proceeding.

## 1. Select CUDA version

Ask the user which public CUDA image variant to build:

- `cu13.0` (default) — builds from the canonical `cu130` Docker stages
- `cu12.8` — builds from the canonical `cu128` Docker stages

Set the public tag suffix and corresponding Docker build target:

```bash
CUDA_TAG_SUFFIX=${CUDA_TAG_SUFFIX:-cu13.0}
case "${CUDA_TAG_SUFFIX}" in
  cu13.0) CUDA_BUILD_TARGET=cu130 ;;
  cu12.8) CUDA_BUILD_TARGET=cu128 ;;
  *) echo "Unsupported CUDA image variant: ${CUDA_TAG_SUFFIX}" >&2; exit 1 ;;
esac
```

Do not pass `cu13.0` or `cu12.8` directly to `docker/Dockerfile`. Its
canonical stage names are `cu130` and `cu128`.

## 2. Determine image tag

```bash
BRANCH=$(git branch --show-current | sed 's/[^a-zA-Z0-9._-]/-/g')
SHA=$(git rev-parse --short=7 HEAD)
TAG="${BRANCH}-${SHA}-${CUDA_TAG_SUFFIX}"
IMAGE="${LMDEPLOY_REGISTRY}/ailab-puyu-puyu_gpu/lmdeploy-dev:${TAG}"
```

Print the computed image name so the user can confirm.

## 3. Build

Ask the user which build mode:

- **patch** (default) — uses `docker/Dockerfile_patch`, fast overlay on existing image
- **full** — uses `docker/Dockerfile`, full multi-stage build from scratch

### Patch build (default)

```bash
BASE_IMAGE=${BASE_IMAGE:-openmmlab/lmdeploy:latest-${CUDA_TAG_SUFFIX}}
docker build -f docker/Dockerfile_patch \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg BACKEND=pytorch \
  --build-arg http_proxy=${http_proxy:-} \
  --build-arg https_proxy=${https_proxy:-} \
  --build-arg no_proxy=${no_proxy:-} \
  -t "${IMAGE}" \
  .
```

User can override:

- `BASE_IMAGE` — defaults to `openmmlab/lmdeploy:latest-${CUDA_TAG_SUFFIX}`
- `BACKEND` — default `pytorch`; set to `turbomind` to include TurboMind C++ extension

### Full build

```bash
docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION="${CUDA_BUILD_TARGET}" \
  --build-arg http_proxy=${http_proxy:-} \
  --build-arg https_proxy=${https_proxy:-} \
  --build-arg no_proxy=${no_proxy:-} \
  -t "${IMAGE}" \
  .
```

`CUDA_BUILD_TARGET` must be the canonical target mapped from the selected
public CUDA variant.

### Verify

```bash
docker images "${IMAGE}"
```

## 4. Push

Skip this step if the user only wants a local build.

### Login

```bash
echo "${REGISTRY_PASSWORD}" | docker login "${LMDEPLOY_REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
```

### Push

```bash
docker push "${IMAGE}"
```

Confirm success via exit code.
