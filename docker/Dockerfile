#docker build -t  openmmlab/lmdeploy-builder:cuda12.2 . -f docker/Dockerfile
ARG CUDA_VERSION=cu12

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cu12
ENV CUDA_VERSION_SHORT=cu121

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cu11
ENV CUDA_VERSION_SHORT=cu118

FROM ${CUDA_VERSION} AS final

ARG PYTHON_VERSION=3.10

ARG TORCH_VERSION=2.3.0
ARG TORCHVISION_VERSION=0.18.0

RUN apt-get update -y && apt-get install -y software-properties-common wget vim git &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update -y && apt-get install -y --no-install-recommends \
    ninja-build rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* && cd /opt && python3 -m venv py3

ENV PATH=/opt/py3/bin:$PATH

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1 &&\
    python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT} &&\
    python3 -m pip install --no-cache-dir cmake packaging wheel &&\
    rm -rf ~/.cache

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz &&\
    tar xf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5 && ./configure --prefix=/usr/local/openmpi &&\
    make -j$(nproc) && make install && cd .. && rm -rf openmpi-4.1.5*

ENV PATH=$PATH:/usr/local/openmpi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib

ENV NCCL_LAUNCH_MODE=GROUP

#install nccl
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt update && \
    apt install libnccl2 libnccl-dev

# Should be in the lmdeploy root directory when building docker image
COPY . /opt/lmdeploy

WORKDIR /opt/lmdeploy

RUN cd /opt/lmdeploy &&\
    python3 -m pip install --no-cache-dir -r requirements.txt &&\
    mkdir -p build && cd build &&\
    sh ../generate.sh &&\
    ninja -j$(nproc) && ninja install &&\
    cd .. &&\
    python3 -m pip install -e . &&\
    rm -rf build  &&\
    rm -rf ~/.cache

ENV LD_LIBRARY_PATH=/opt/lmdeploy/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/lmdeploy/install/bin:$PATH

# explicitly set ptxas path for triton
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
