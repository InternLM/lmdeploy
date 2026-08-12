# LMDeploy Build System

## Building lmdeploy builder images

Build the default CUDA 12.8 builder image:

```bash
./build_all_lmdeploy_builders.sh

# Build and push the image (for CI/CD)
WITH_PUSH=true ./build_all_lmdeploy_builders.sh
```

Set `CUDA_VERSION` to build a different version, or use a space-separated list
to build multiple versions explicitly:

```bash
CUDA_VERSION=13.0 ./build_all_lmdeploy_builders.sh
CUDA_VERSION='12.8 13.0' WITH_PUSH=true ./build_all_lmdeploy_builders.sh
```

For custom builds with specific manylinux and CUDA versions:

```bash
MANY_LINUX_VERSION=2_28 GPU_ARCH_VERSION=13.0 ./build_lmdeploy_builder.sh
```

## Build lmdeploy wheels

Compile all wheel packages:

```bash
./build_all_wheel.sh

# Build CUDA 13.0 wheels
CUDA_VER=13.0 ./build_all_wheel.sh
```
