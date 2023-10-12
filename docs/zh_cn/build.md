### 源码安装

- 安装编译和运行依赖包：

  ```shell
  conda create -n lmdeploy python=3.10
  conda activate lmdeploy

  git clone https://github.com/InternLM/lmdeploy.git
  cd lmdeploy

  pip install -r requirements.txt
  conda install openmpi-mpicxx nccl rapidjson -c conda-forge
  ```

- lmdeploy 编译安装:

  ```shell
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
