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
  wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.0.tar.gz
  tar -xzf openmpi-*.tar.gz && cd openmpi-*
  ./configure --with-cuda
  make -j$(nproc)
  make install
  ```
- build and install lmdeploy:
  ```shell
  mkdir build && cd build
  sh ../generate.sh
  ```

Then, you can communicate with the inference server by command line,

```shell
python3 -m lmdeploy.turbomind.chat model_path
```

or webui,

```shell
python3 -m lmdeploy.app model_path
```
