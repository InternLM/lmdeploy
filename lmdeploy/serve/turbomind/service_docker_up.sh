#!/bin/sh

show_help() {
  echo "Usage: $0 [-h] [--help] [-l] [--lib-dir]"
  echo
  echo "Options:"
  echo "  -h, --help   Show this help message and exit"
  echo "  --lib-dir    Specify the directory of turbomind libraries"
}

# check if '-h' or '--help' in the arguments
for arg in "$@"
do
  if [ "$arg" == "-h" ] || [ "$arg" == "--help" ]; then
    show_help
    exit 0
  fi
done


TP=1
DEVICES="0"
for ((i = 1; i < ${TP}; ++i)); do
    DEVICES="${DEVICES},$i"
done
DEVICES="\"device=${DEVICES}\""


SCRIPT_DIR="$(dirname "$0")"
SCRIPT_ABS_DIR="$(realpath "$SCRIPT_DIR")"


if [ -z "$1" ]; then
    docker run \
        --gpus $DEVICES \
        --rm \
        -v "${SCRIPT_ABS_DIR}":/workspace/models \
        --shm-size 16g \
        -p 33336:22 \
        -p 33337-33400:33337-33400 \
        --cap-add=SYS_PTRACE \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        --name lmdeploy \
        -it --env NCCL_LAUNCH_MODE=GROUP openmmlab/lmdeploy:latest \
        tritonserver \
        --model-repository=/workspace/models/model_repository \
        --allow-http=0 \
        --allow-grpc=1 \
        --grpc-port=33337 \
        --log-verbose=0 \
        --allow-metrics=1
fi

for ((i = 1; i <= $#; i++)); do
  arg=${!i}
  case "$arg" in
    --lib-dir)
    if [ "$i" -eq "$#" ]; then
        show_help
        exit -1
    fi
    LIB_PATH=${@:i+1:1}
      docker run \
        --gpus $DEVICES \
        --rm \
        -v "${LIB_PATH}":/opt/tritonserver/backends/turbomind \
        -v ""${SCRIPT_ABS_DIR}"":/workspace/models \
        --shm-size 16g \
        -p 33336:22 \
        -p 33337-33400:33337-33400 \
        --cap-add=SYS_PTRACE \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        --name lmdeploy \
        -it --env NCCL_LAUNCH_MODE=GROUP openmmlab/lmdeploy:latest \
        tritonserver \
        --model-repository=/workspace/models/model_repository \
        --allow-http=0 \
        --allow-grpc=1 \
        --grpc-port=33337 \
        --log-verbose=0 \
        --allow-metrics=1
    break
    ;;
  esac
done
