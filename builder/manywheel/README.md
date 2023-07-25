# lmdeploy/builder/manywheel

## Building docker images

To build all docker images you can use the convenience script:

```bash
./build_all_docker.sh
# Build with pushing
WITH_PUSH=true ./build_all_docker.sh
```

To build a specific docker image use:

```bash
MANY_LINUX_VERSION=2014 GPU_ARCH_VERSION=11.8 ./build_docker.sh
```

## Building lmdeploy

```bash
./build_all_wheel.sh
```
