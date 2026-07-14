#!/bin/bash

set -ex

function install_118 {
    echo "Installing CUDA 11.8 and NCCL 2.15"
    rm -rf /usr/local/cuda-11.8 /usr/local/cuda
    # install CUDA 11.8.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
    rm -f cuda_11.8.0_520.61.05_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.8 /usr/local/cuda

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
    echo "Installing CUDA 12.1 and NCCL 2.18.1"
    rm -rf /usr/local/cuda-12.1 /usr/local/cuda
    # install CUDA 12.1.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    chmod +x cuda_12.1.0_530.30.02_linux.run
    ./cuda_12.1.0_530.30.02_linux.run --toolkit --silent
    rm -f cuda_12.1.0_530.30.02_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.1 /usr/local/cuda

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

function install_124 {
    echo "Installing CUDA 12.4 and NCCL 2.25.1"
    rm -rf /usr/local/cuda-12.4 /usr/local/cuda
    # install CUDA 12.4.1 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    chmod +x cuda_12.4.1_550.54.15_linux.run
    ./cuda_12.4.1_550.54.15_linux.run --toolkit --silent
    rm -f cuda_12.4.1_550.54.15_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.4 /usr/local/cuda

    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    mkdir tmp_nccl && cd tmp_nccl
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.25.1/nccl_2.25.1-1+cuda12.4_x86_64.txz
    tar xf nccl_2.25.1-1+cuda12.4_x86_64.txz
    cp -a nccl_2.25.1-1+cuda12.4_x86_64/include/* /usr/local/cuda/include/
    cp -a nccl_2.25.1-1+cuda12.4_x86_64/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_nccl
    ldconfig
}

function install_126 {
    echo "Installing CUDA 12.6 and NCCL 2.24.3"
    rm -rf /usr/local/cuda-12.6 /usr/local/cuda
    # install CUDA 12.6.3 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
    chmod +x cuda_12.6.3_560.35.05_linux.run
    ./cuda_12.6.3_560.35.05_linux.run --toolkit --silent
    rm -f cuda_12.6.3_560.35.05_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.6 /usr/local/cuda

    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    mkdir tmp_nccl && cd tmp_nccl
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.24.3/nccl_2.24.3-1+cuda12.6_x86_64.txz
    tar xf nccl_2.24.3-1+cuda12.6_x86_64.txz
    cp -a nccl_2.24.3-1+cuda12.6_x86_64/include/* /usr/local/cuda/include/
    cp -a nccl_2.24.3-1+cuda12.6_x86_64/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_nccl
    ldconfig
}

function install_128 {
    echo "Installing CUDA 12.8 and NCCL 2.25.1"
    rm -rf /usr/local/cuda-12.8 /usr/local/cuda
    # install CUDA 12.8.1 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent
    rm -f cuda_12.8.1_570.124.06_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.8 /usr/local/cuda

    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    mkdir tmp_nccl && cd tmp_nccl
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.25.1/nccl_2.25.1-1+cuda12.8_x86_64.txz
    tar xf nccl_2.25.1-1+cuda12.8_x86_64.txz
    cp -a nccl_2.25.1-1+cuda12.8_x86_64/include/* /usr/local/cuda/include/
    cp -a nccl_2.25.1-1+cuda12.8_x86_64/lib/* /usr/local/cuda/lib64/
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
    12.4) install_124
            ;;
    12.6) install_126
            ;;
    12.8) install_128
            ;;
	*) echo "bad argument $1"; exit 1
	   ;;
    esac
    shift
done
