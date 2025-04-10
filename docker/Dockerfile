ARG CUDA_VERSION=cu12

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu121

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cu11
ENV CUDA_VERSION_SHORT=cu118

FROM ${CUDA_VERSION} AS final

ARG PYTHON_VERSION=3.10

RUN apt-get update -y && apt-get install -y software-properties-common wget vim git curl openssh-server ssh sudo &&\
    apt-get install libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 -y &&\
    apt-get install libibverbs-dev rdma-core -y &&\
    curl https://sh.rustup.rs -sSf | sh -s -- -y &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update -y && apt-get install -y --no-install-recommends \
    ninja-build rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* && cd /opt && python3 -m venv py3

ENV PATH=/opt/py3/bin:$PATH

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz &&\
    tar xf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5 && ./configure --prefix=/usr/local/openmpi &&\
    make -j$(nproc) && make install && cd .. && rm -rf openmpi-4.1.5*

ENV PATH=$PATH:/usr/local/openmpi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib

# install nccl manually for cuda11.8
RUN if [ "$CUDA_VERSION_SHORT" = "cu118" ]; then \
    git clone --depth=1 --branch v2.22.3-1 https://github.com/NVIDIA/nccl.git &&\
    cd nccl && make -j$(nproc) src.build &&\
    mv build/include/* /usr/local/include &&\
    mkdir -p /usr/local/nccl/lib &&\
    mv build/lib/lib* /usr/local/nccl/lib/ &&\
    cd .. && rm -rf nccl ; \
    fi

ENV LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH


RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip setuptools==69.5.1 wheel &&\
    python3 -m pip install cmake packaging wheel

ENV NCCL_LAUNCH_MODE=GROUP

# Should be in the lmdeploy root directory when building docker image
COPY . /opt/lmdeploy

WORKDIR /opt/lmdeploy

RUN --mount=type=cache,target=/root/.cache/pip cd /opt/lmdeploy &&\
    python3 -m pip install -r requirements_cuda.txt --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT} &&\
    mkdir -p build && cd build &&\
    sh ../generate.sh &&\
    ninja -j$(nproc) && ninja install &&\
    cd .. &&\
    python3 -m pip install -e . &&\
    rm -rf build

# use locally built nccl for cuda11.8
RUN if [ "$CUDA_VERSION_SHORT" = "cu118" ]; then python3 -m pip uninstall -y nvidia-nccl-cu11 ; fi

ENV LD_LIBRARY_PATH=/opt/lmdeploy/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/lmdeploy/install/bin:$PATH

# explicitly set ptxas path for triton
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# GDRCopy
WORKDIR /tmp
RUN dpkg -r libgdrapi gdrcopy && \
    wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz &&\
    tar -zxvf v2.4.4.tar.gz && cd gdrcopy-2.4.4 && \
    make prefix=/usr/local/gdrcopy install -j$(nproc) &&\
    rm -rf /tmp/* /var/tmp/*

# DeepEP
WORKDIR /opt
RUN git clone https://github.com/deepseek-ai/DeepEP.git

# NVSHMEM
WORKDIR /opt
ENV CUDA_HOME=/usr/local/cuda
RUN wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz &&\
    tar xvf nvshmem_src_3.2.5-1.txz &&\
    cd nvshmem_src && git apply /opt/DeepEP/third-party/nvshmem.patch &&\
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/usr/local/nvshmem \
    -DMLX5_lib=/lib/x86_64-linux-gnu/libmlx5.so.1 &&\
    cmake --build build --target install --parallel $(nproc) &&\
    cd .. && rm nvshmem_src_3.2.5-1.txz && rm -rf nvshmem_src

# install DeepEP
WORKDIR /opt
RUN cd DeepEP && NVSHMEM_DIR=/usr/local/nvshmem pip install -v . &&\
    cd .. && rm -rf DeepEP

# set workspace
WORKDIR /opt/lmdeploy
