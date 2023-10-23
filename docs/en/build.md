# Build from source

LMDeploy provides prebuilt package that can be easily installed by `pip install lmdeploy`.

If you have requests to build lmdeploy from source, please read the following sections.

## Build in Docker (recommended)

We highly advise using the provided docker image for lmdeploy build to circumvent complex environment setup

```shell
# clone lmdeploy source code and its docker image
git clone --depth=1 https://github.com/InternLM/lmdeploy
docker pull openmmlab/lmdeploy:latest

# launch docker container
cd {home/folder/of/lmdeploy}
docker run --gpus all --rm -v $(pwd):/workspace/lmdeploy -it openmmlab/lmdeploy:latest /bin/bash

# build and install
cd /workspace/lmdeploy
mkdir -p build && cd build
../generate.sh
make -j$(nproc) && make install
```

**Note**:

- Due to the need to compile kernels under multiple architectures, the compilation time is quite long.
- Apart from mapping source code from the host machine to the docker image using data volumes, you can also directly use the lmdeploy source code within the docker image.
  Its path is `/opt/tritonserver/lmdeploy`.

## Build in localhost (optional)

Firstly, please make sure gcc version is no less than 9, which can be conformed by `gcc --version`.

Then, follow the steps below to set up the compilation environment:

- install the dependent packages:
  ```shell
  pip install -r requirements.txt
  apt-get install rapidjson-dev
  ```
- install [nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html), and set environment variables:
  ```shell
  export NCCL_ROOT_DIR=/path/to/nccl/build
  export NCCL_LIBRARIES=/path/to/nccl/build/lib
  ```
- install openmpi from source:
  ```shell
  wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
  tar -xzf openmpi-*.tar.gz && cd openmpi-*
  ./configure --with-cuda
  make -j$(nproc)
  make install
  ```
- build and install lmdeploy:
  ```shell
  cd {home/folder/of/lmdeploy}
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
