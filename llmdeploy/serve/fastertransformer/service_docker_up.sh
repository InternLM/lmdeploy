#!/bin/sh

docker run \
    --gpus "device=0" \
    --rm \
    -v /home/lvhan/nvidia/mmdeploy-fastertransformer-backend/build/lib:/opt/tritonserver/backends/fastertransformer \
    -v $(pwd):/workspace/models \
    --shm-size 16g \
    -p 33336:22 \
    -p 33337-33400:33337-33400 \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --name llmdeploy \
    -it --env NCCL_LAUNCH_MODE=GROUP lvhan028/fastertransformer:v0.0.1 \
    tritonserver \
    --model-repository=/workspace/models/model_repository \
    --allow-http=0 \
    --allow-grpc=1 \
    --grpc-port=33337 \
    --log-verbose=0 \
    --allow-metrics=1
