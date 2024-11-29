ARG JETPACK_VERSION=36.2.0  # Jetpack 6.0 DP

FROM "nvcr.io/nvidia/l4t-jetpack:r$JETPACK_VERSION"

ARG JETPACK_VERSION=36.2.0
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

ENV JETPACK_VERSION=$JETPACK_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends rapidjson-dev libgoogle-glog-dev gdb python3-pip git libopenblas-dev libopenmpi-dev && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1 &&\
    python3 -m pip install --no-cache-dir cmake packaging wheel && \
    mkdir -p /workspace

COPY . /workspace/lmdeploy

WORKDIR /workspace/lmdeploy

RUN mkdir -p /workspace/lmdeploy/torch-jetson && \
    chmod +x /workspace/lmdeploy/docker/jetson/install_pytorch.sh && \
    /workspace/lmdeploy/docker/jetson/install_pytorch.sh $JETPACK_VERSION

RUN cd /workspace/lmdeploy &&\
    python3 -m pip install --no-cache-dir -r requirements/build.txt

# proxy
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN mkdir -p build && cd build &&\
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_INSTALL_PREFIX=/workspace/lmdeploy/install \
        -DBUILD_PY_FFI=ON \
        -DBUILD_MULTI_GPU=OFF \
        -DBUILD_CUTLASS_MOE=OFF \
        -DBUILD_CUTLASS_MIXED_GEMM=OFF \
        -DCMAKE_CUDA_FLAGS="-lineinfo" \
        -DUSE_NVTX=ON &&\
    make -j$(nproc) && make install &&\
    cd .. &&\
    python3 -m pip install -e . &&\
    rm -rf build

ENV LD_LIBRARY_PATH=/workspace/lmdeploy/install/lib:$LD_LIBRARY_PATH
# # explicitly set ptxas path for triton
# ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV http_proxy=""
ENV https_proxy=""
