## Build from source

- install packages for compiling and running:

  ```shell
  conda create -n lmdeploy python=3.10
  conda activate lmdeploy

  git clone https://github.com/InternLM/lmdeploy.git
  cd lmdeploy

  pip install -r requirements.txt
  conda install openmpi-mpicxx nccl rapidjson -c conda-forge
  ```

- build and install lmdeploy:

  ```shell
  mkdir build && cd build
  sh ../generate.sh
  make -j$(nproc) && make install
  ```
