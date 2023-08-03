#!/bin/bash

set -ex

function install_118 {
    echo "Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15"
    rm -rf /usr/local/cuda-11.8 /usr/local/cuda
    # install CUDA 11.8.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
    rm -f cuda_11.8.0_520.61.05_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.8 /usr/local/cuda

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
    tar xf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
    cp -a cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig

    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    mkdir tmp_nccl && cd tmp_nccl
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.15.5/nccl_2.15.5-1+cuda11.8_x86_64.txz
    tar xf nccl_2.15.5-1+cuda11.8_x86_64.txz
    cp -a nccl_2.15.5-1+cuda11.8_x86_64/include/* /usr/local/cuda/include/
    cp -a nccl_2.15.5-1+cuda11.8_x86_64/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_nccl
    ldconfig
}

function install_121 {
    echo "Installing CUDA 12.1 and cuDNN 8.9 and NCCL 2.18.1"
    rm -rf /usr/local/cuda-12.1 /usr/local/cuda
    # install CUDA 12.1.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    chmod +x cuda_12.1.0_530.30.02_linux.run
    ./cuda_12.1.0_530.30.02_linux.run --toolkit --silent
    rm -f cuda_12.1.0_530.30.02_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.1 /usr/local/cuda

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz -O cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-8.9.2.26_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-8.9.2.26_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig

    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    mkdir tmp_nccl && cd tmp_nccl
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.18.1/nccl_2.18.1-1+cuda12.1_x86_64.txz
    tar xf nccl_2.18.1-1+cuda12.1_x86_64.txz
    cp -a nccl_2.18.1-1+cuda12.1_x86_64/include/* /usr/local/cuda/include/
    cp -a nccl_2.18.1-1+cuda12.1_x86_64/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_nccl
    ldconfig
}

if test $# -eq 0
then
    echo "doesn't provide cuda version"; exit 1;
fi

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    11.8) install_118
	        ;;
    12.1) install_121
            ;;
	*) echo "bad argument $1"; exit 1
	   ;;
    esac
    shift
done
