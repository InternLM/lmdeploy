# 编译和安装

LMDeploy 提供了预编译包，可以很方便的通过 `pip install lmdeploy` 安装和使用。

如果有源码编译的需求，请先下载 lmdeploy 源码：

```shell
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

然后，参考以下章节编译和安装。

## 在 docker 内编译安装（强烈推荐）

LMDeploy 提供了编译镜像 `openmmlab/lmdeploy-builder:cuda11.8` ~~和运行镜像 `openmmlab/lmdeploy:latest`。~~

~~前者用来编译 lmdeploy 并打包为 wheel 文件。后者用来运行 lmdeploy，对模型进行推理。~~

只需运行以下命令，即可在 bui，得到 lmdeploy 在 py3.8 - py3.11 下所有的 wheel 文件。

```shell
cd lmdeploy # lmdeploy 源码根目录
bash builder/manywheel/build_all_wheel.sh
```

如果需要固定 python 版本的 wheel 文件，比如 py3.8，可以执行：

```shell
bash builder/manywheel/build_wheel.sh py38 manylinux2014_x86_64 cuda11.8 cuda11.8_dist
```

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
  tar xf openmpi-4.1.5.tar.gz
  cd openmpi-4.1.5
  ./configure
  make -j$(nproc) && make install
  ```
- lmdeploy 编译安装:
  ```shell
  cd lmdeploy # lmdeploy 源码的根目录
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
- 安装 lmdeploy python package:
  ```shell
  cd ..
  pip install -e .
  ```
