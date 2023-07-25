# lmdeploy/builder/manywheel

## Building docker images

To build all docker images you can use the convenience script:

```bash
manywheel/build_all_docker.sh
# Build with pushing
WITH_PUSH=true manywheel/build_all_docker.sh
```

To build a specific docker image use:
```bash
# GPU_ARCH_TYPE can be ["cuda", "rocm", "cpu"]
# GPU_ARCH_VERSION is GPU_ARCH_TYPE dependent, see manywheel/build_all_docker.sh for examples
GPU_ARCH_TYPE=cuda GPU_ARCH_VERSION=11.1 manywheel/build_docker.sh
# Build with pushing
WITH_PUSH=true GPU_ARCH_TYPE=cuda GPU_ARCH_VERSION=11.1 manywheel/build_docker.sh
```

**DISCLAIMER for WITH_PUSH**:
If you'd like to use WITH_PUSH, you must set it to exactly `true`, not `1` nor `ON` nor even `TRUE`, as our scripts
check for exact string equality to enable push functionality. The reason for this rigidity is due to the how we
calculate WITH_PUSH in our GHA workflow YAMLs. Currently, we usually enable push based on the workflow trigger, which
when we query with an expression like `${{ github.event_name == 'push' }}` returns either `true` or `false`. Thus, we
adapt our scripts to fit with this model.




/tmp/builder
docker.io/pytorch/manylinux-builder:cuda11.8
--build-arg BASE_CUDA_VERSION=11.8 --build-arg DEVTOOLSET_VERSION=9
centos:7
cuda_final