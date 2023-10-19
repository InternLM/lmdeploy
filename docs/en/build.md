## Build from source

- make sure local gcc version no less than 9, which can be conformed by `gcc --version`.
- install packages for compiling and running:
  ```shell
  pip install -r requirements.txt
  ```
- install [nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html), set environment variables:
  ```shell
  export NCCL_ROOT_DIR=/path/to/nccl/build
  export NCCL_LIBRARIES=/path/to/nccl/build/lib
  ```
- install rapidjson
- install openmpi, installing from source is recommended.
  ```shell
  wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
  tar -xzf openmpi-*.tar.gz && cd openmpi-*
  ./configure --with-cuda
  make -j$(nproc)
  make install
  ```
- build and install lmdeploy:
  ```shell
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
