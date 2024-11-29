#!/bin/bash

JETPACK_VERSION=$1

echo $JETPACK_VERSION

if [ "$JETPACK_VERSION" = "35.2.1" ] ; then     # Jetpack 5.1
    wget "https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl" -P /workspace/lmdeploy/torch-jetson
    python3 -m pip install "/workspace/lmdeploy/torch-jetson/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl"
elif [ "$JETPACK_VERSION" = "35.3.1" ] ; then   # Jetpack 5.1.1
    wget "https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0a0+fe05266f.nv23.04-cp38-cp38-linux_aarch64.whl" -P /workspace/lmdeploy/torch-jetson
    python3 -m pip install "/workspace/lmdeploy/torch-jetson/torch-2.0.0a0+fe05266f.nv23.04-cp38-cp38-linux_aarch64.whl"
elif [ "$JETPACK_VERSION" = "35.4.1" ] ; then   # Jetpack 5.1.2
    wget "https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl" -P /workspace/lmdeploy/torch-jetson
    python3 -m pip install "/workspace/lmdeploy/torch-jetson/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
elif [ "$JETPACK_VERSION" = "36.2.0" ] ; then   # Jetpack 6.0 DP
    wget "https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl" -P /workspace/lmdeploy/torch-jetson
    python3 -m pip install "/workspace/lmdeploy/torch-jetson/torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl"
elif [ "$JETPACK_VERSION" = "36.3.0" ] ; then   # Jetpack 6.0
    wget https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -P /workspace/lmdeploy/torch-jetson
    python3 -m pip install "/workspace/lmdeploy/torch-jetson/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl"
else
    echo "We currently do not support Jetpack v$JETPACK_VERSION. Please try 35.2.1, 35.3.1, 35.4.1, 36.2.0, or 36.3.0"
    exit 1
fi
