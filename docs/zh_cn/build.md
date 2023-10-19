### 源码安装

- 确保物理机环境的 gcc 版本不低于 9，可以通过`gcc --version`确认。
- 安装编译和运行依赖包：
  ```shell
  pip install -r requirements.txt
  ```
- 安装 [nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html),设置环境变量
  ```shell
  export NCCL_ROOT_DIR=/path/to/nccl/build
  export NCCL_LIBRARIES=/path/to/nccl/build/lib
  ```
- rapidjson 安装
- openmpi 安装, 推荐从源码安装:
  ```shell
  wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
  tar -xzf openmpi-*.tar.gz && cd openmpi-*
  ./configure --with-cuda
  make -j$(nproc)
  make install
  ```
- lmdeploy 编译安装:
  ```shell
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
