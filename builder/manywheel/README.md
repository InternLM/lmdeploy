# Build lmdeploy manylinux wheel

## Prepare docker image

To build all docker images you can use the convenient script:

```bash
./build_all_docker.sh
# Build with pushing
WITH_PUSH=true ./build_all_docker.sh
```

To build a docker image with specific cuda version or manylinux-docker version, you may use:

```bash
MANY_LINUX_VERSION=2014 GPU_ARCH_VERSION=11.8 ./build_docker.sh
```

## Build lmdeploy wheel

```bash
./build_all_wheel.sh
```
