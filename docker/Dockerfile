ARG CUDA_VERSION=cu12

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu121

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cu11
ENV CUDA_VERSION_SHORT=cu118

FROM ${CUDA_VERSION} AS final

ARG PYTHON_VERSION=3.10

ARG TORCH_VERSION=2.3.0
ARG TORCHVISION_VERSION=0.18.0

RUN rm /etc/apt/sources.list.d/cuda*.list && apt-get update -y && apt-get install -y software-properties-common wget vim &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update -y && apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* && cd /opt && python3 -m venv py3

ENV PATH=/opt/py3/bin:$PATH

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1 &&\
    python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT} &&\
    python3 -m pip install --no-cache-dir cmake packaging wheel

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz &&\
    tar xf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5 && ./configure --prefix=/usr/local/openmpi &&\
    make -j$(nproc) && make install && rm -rf openmpi-4.1.5*

ENV PATH=$PATH:/usr/local/openmpi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib

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
        -DCMAKE_INSTALL_PREFIX=/opt/lmdeploy/install \
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

ENV LD_LIBRARY_PATH=/opt/lmdeploy/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/lmdeploy/install/bin:$PATH

# explicitly set ptxas path for triton
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
