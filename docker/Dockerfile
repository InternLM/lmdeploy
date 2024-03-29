FROM nvcr.io/nvidia/tritonserver:24.03-py3

RUN rm /etc/apt/sources.list.d/cuda*.list && apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb python3.10-venv \
    && rm -rf /var/lib/apt/lists/* && cd /opt && python3 -m venv py310

ENV PATH=/opt/py310/bin:$PATH

RUN python3 -m pip install --no-cache-dir --upgrade pip &&\
    python3 -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 &&\
    python3 -m pip install --no-cache-dir cmake packaging wheel

ENV NCCL_LAUNCH_MODE=GROUP

# Should be in the lmdeploy root directory when building docker image
COPY . /opt/lmdeploy

WORKDIR /opt/lmdeploy

RUN cd /opt/lmdeploy &&\
    python3 -m pip install --no-cache-dir -r requirements.txt &&\
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
    python3 -m pip install -e . &&\
    rm -rf build

ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
