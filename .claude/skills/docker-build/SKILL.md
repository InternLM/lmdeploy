---
name: docker-build
description: Build an LMDeploy Docker image and push it to the inner registry.
---

# Docker Build & Push

Build an LMDeploy Docker image and push it to the inner registry.

## Prerequisites

Before starting, verify all three environment variables are set:

```bash
echo $LMDEPLOY_REGISTRY    # inner registry hostname/path
echo $REGISTRY_USER        # registry login username
echo $REGISTRY_PASSWORD    # registry login password
```

If any are missing, stop and tell the user to set them before proceeding.

## 1. Determine image tag

```bash
BRANCH=$(git branch --show-current | sed 's/[^a-zA-Z0-9._-]/-/g')
SHA=$(git rev-parse --short=7 HEAD)
TAG="${BRANCH}-${SHA}"
IMAGE="${LMDEPLOY_REGISTRY}/lmdeploy:${TAG}"
```

Print the computed image name so the user can confirm.

## 2. Build

Ask the user which build mode:

- **patch** (default) — uses `docker/Dockerfile_patch`, fast overlay on existing image
- **full** — uses `docker/Dockerfile`, full multi-stage build from scratch

### Patch build (default)

```bash
docker build -f docker/Dockerfile_patch \
  --build-arg BASE_IMAGE=openmmlab/lmdeploy:v0.12.3.post2-cu12.8 \
  --build-arg BACKEND=pytorch \
  --build-arg http_proxy=${http_proxy:-} \
  --build-arg https_proxy=${https_proxy:-} \
  --build-arg no_proxy=${no_proxy:-} \
  -t "${IMAGE}" \
  .
```

User can override:

- `BASE_IMAGE` — default `openmmlab/lmdeploy:v0.12.3.post2-cu12.8`
- `BACKEND` — default `pytorch`; set to `turbomind` to include TurboMind C++ extension

### Full build

```bash
docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION=cu12.8 \
  --build-arg http_proxy=${http_proxy:-} \
  --build-arg https_proxy=${https_proxy:-} \
  --build-arg no_proxy=${no_proxy:-} \
  -t "${IMAGE}" \
  .
```

User can override `CUDA_VERSION` — default `cu12.8`.

### Verify

```bash
docker images "${IMAGE}"
```

## 3. Push

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
