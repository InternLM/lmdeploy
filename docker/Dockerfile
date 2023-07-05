FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb  \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install sentencepiece cmake transformers protobuf==3.20.3 pybind11 mmengine

ENV NCCL_LAUNCH_MODE=GROUP

RUN git clone --depth=1 https://github.com/open-mmlab/lmdeploy.git &&\
    cd lmdeploy &&\
    mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DBUILD_MULTI_GPU=ON \
        -DBUILD_CUTLASS_MOE=OFF \
        -DBUILD_CUTLASS_MIXED_GEMM=OFF \
        -DCMAKE_CUDA_FLAGS="-lineinfo" \
        -DUSE_NVTX=ON && \
    make -j$(nproc) && make install &&\
    cd .. &&\
    rm -rf build

ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
