FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb  \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install cmake protobuf==3.20.3

ENV NCCL_LAUNCH_MODE=GROUP

ARG COMMIT_ID=main

RUN git clone --depth=1 https://github.com/InternLM/lmdeploy.git &&\
    cd lmdeploy &&\
    git checkout ${COMMIT_ID} &&\
    python3 -m pip install -r requirements.txt &&\
    mkdir -p build && cd build &&\
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_INSTALL_PREFIX=/opt/tritonserver \
        -DBUILD_PY_FFI=ON \
        -DBUILD_MULTI_GPU=ON \
        -DBUILD_CUTLASS_MOE=OFF \
        -DBUILD_CUTLASS_MIXED_GEMM=OFF \
        -DCMAKE_CUDA_FLAGS="-lineinfo" \
        -DUSE_NVTX=ON &&\
    make -j$(nproc) && make install &&\
    cd .. &&\
    python3 -m pip install . &&\
    rm -rf build

ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
