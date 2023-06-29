FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb  \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install sentencepiece cmake transformers protobuf==3.20.3

ENV NCCL_LAUNCH_MODE=GROUP
