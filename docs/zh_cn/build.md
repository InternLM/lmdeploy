# 编译和安装

LMDeploy 提供了预编译包，可以很方便的通过 `pip install lmdeploy` 安装和使用。

如果有源码编译的需求，请参考以下章节。

## 在 docker 内编译安装（强烈推荐）

使用 lmdeploy 提供的 docker 镜像编译源码，可以避免复杂的环境配置。我们强烈推荐您使用这种方式。

```shell
# 下载 lmdeploy 源码，以及编译镜像
git clone --depth=1 https://github.com/InternLM/lmdeploy
docker pull openmmlab/lmdeploy:latest

# 启动 docker container
cd {home/folder/of/lmdeploy}
docker run --gpus all --rm -v $(pwd):/workspace/lmdeploy -it openmmlab/lmdeploy:latest /bin/bash

# 编译和安装
cd /workspace/lmdeploy
mkdir -p build && cd build
../generate.sh
make -j$(nproc) && make install
```

**说明**

- 因为要编译很多架构下的kernel，所以编译时间比较长。
- 除了通过上面的数据卷方式，把源码从宿主机映射到docker image，您也可以直接使用镜像中 lmdeploy 的源码。它的路径是 `/opt/tritonserver/lmdeploy`。

## 在 localhost 编译安装（可选）

首先，请确保物理机环境的 gcc 版本不低于 9，可以通过`gcc --version`确认。

然后，按如下步骤，配置编译环境：

- 安装编译和运行依赖包：
  ```shell
  pip install -r requirements.txt
  apt-get install rapidjson-dev
  ```
- 安装 [nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html),设置环境变量
  ```shell
  export NCCL_ROOT_DIR=/path/to/nccl/build
  export NCCL_LIBRARIES=/path/to/nccl/build/lib
  ```
- 源码编译安装 openmpi:
  ```shell
  wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
  tar -xzf openmpi-*.tar.gz && cd openmpi-*
  ./configure --with-cuda
  make -j$(nproc)
  make install
  ```
- lmdeploy 编译安装:
  ```shell
  cd {home/folder/of/lmdeploy}
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
