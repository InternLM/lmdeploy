# LMDeploy Build System

## Building lmdeploy builder images

To build all lmdeploy builder images, such as "lmdeploy-builder:cuda11.8", ""lmdeploy-builder:cuda12.4", execute:

```bash
./build_all_lmdeploy_builders.sh

# Build and push images (for CI/CD)
WITH_PUSH=true ./build_all_lmdeploy_builders.sh
```

For custom builds with specific versions:

```bash
MANY_LINUX_VERSION=2014 GPU_ARCH_VERSION=12.4 ./build_lmdeploy_builder.sh
```

## Build lmdeploy wheels

Compile all wheel packages:

```bash
./build_all_wheel.sh
```
