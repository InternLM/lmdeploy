# Build from source

LMDeploy provides prebuilt package that can be easily installed by `pip install lmdeploy`.

If you have requests to build lmdeploy from source, please clone lmdeploy repository from GitHub, and follow instructions in next sections

```shell
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

## Build in Docker (recommended)

We highly advise using the provided docker image for lmdeploy build to circumvent complex environment setup.

The docker image is `openmmlab/lmdeploy-builder:cuda11.8`. Make sure that docker is installed before using this image.

In the root directory of the lmdeploy source code, please run the following command:

```shell
# the home folder of lmdeploy source code
cd lmdeploy
bash builder/manywheel/build_all_wheel.sh
```

All the wheel files for lmdeploy under py3.8 - py3.11 will be found in the `builder/manywheel/cuda11.8_dist` directory, such as,

```text
builder/manywheel/cuda11.8_dist/
├── lmdeploy-0.0.12-cp310-cp310-manylinux2014_x86_64.whl
├── lmdeploy-0.0.12-cp311-cp311-manylinux2014_x86_64.whl
├── lmdeploy-0.0.12-cp38-cp38-manylinux2014_x86_64.whl
└── lmdeploy-0.0.12-cp39-cp39-manylinux2014_x86_64.whl
```

If the wheel file for a specific Python version is required, such as py3.8, please execute:

```shell
bash builder/manywheel/build_wheel.sh py38 manylinux2014_x86_64 cuda11.8 cuda11.8_dist
```

And the wheel file will be found in the `builder/manywheel/cuda11.8_dist` directory.

You can use `pip install` to install the wheel file that matches the Python version on your host machine.

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
  git clone https://github.com/NVIDIA/nccl.git
  cd nccl && make -j src.build
  export NCCL_ROOT_DIR=/path/to/nccl/build
  export NCCL_LIBRARIES=/path/to/nccl/build/lib
  ```
- install openmpi from source:
  ```shell
  wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
  tar xf openmpi-4.1.5.tar.gz
  cd openmpi-4.1.5
  ./configure --prefix=/usr/local/openmpi
  make -j$(nproc) && make install
  export PATH=$PATH:/usr/local/openmpi/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib
  ```
- build and install lmdeploy libraries:
  ```shell
  # install ninja
  apt install ninja-build
  # the home folder of lmdeploy
  cd lmdeploy
  mkdir build && cd build
  sh ../generate.sh
  ninja -j$(nproc) && ninja install
  ```
- install lmdeploy python package:
  ```shell
  cd ..
  pip install -e .
  ```
