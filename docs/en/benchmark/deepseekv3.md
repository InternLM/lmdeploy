# DeepSeekV3 Support

LMdeploy has now supported DeepSeekV3! :rocket: :rocket: :rocket:

In this document, we will provide step-by-step guidance to set up DeepSeekV3 inference with LMDeploy on a multi-node cluster.

## 1. Installation using docker

We highly recommend users adopt our official docker image to avoid potential errors caused by the environment differences.

### 1.1 Pull docker image

To start with, pull the latest LMDeploy docker image by

```bash
docker pull openmmlab/lmdeploy:latest-cu11
```

### 1.2 Build docker containers

Execute the following commands on both head and slave nodes to create docker containers.

Modify the `MODEL_PATH`, `CONTAINER_MODEL_PATH`, `DOCKER_IMG_NAME` to your own.

```bash
# The path to the shared model weights folder outside the container.
MODEL_PATH='/mnt/huggingface_hub_137_llm/hub/models--deepseek-ai--DeepSeek-V3'
# The path to the mounted folder inside the container.
CONTAINER_MODEL_PATH='/mnt/huggingface_hub_137_llm/hub/models--deepseek-ai--DeepSeek-V3'

# Docker image name / ID
DOCKER_IMG_NAME='xxx'

# Get all RDMA devices
devices=$(ls /dev/infiniband)
# Build --device parameters
device_args=""
for dev in $devices; do
    device_args+="--device=/dev/infiniband/$dev "
done

sudo docker run -it \
    --gpus all \
    --network host \
    --ipc host \
    --name lmdeploy \
    --ulimit memlock=-1 \
    -v $MODEL_PATH:$CONTAINER_MODEL_PATH \
    $device_args \
    $DOCKER_IMG_NAME
```

Note that `--ipc host` and `$device_args` may largely impact inference efficiency. Ensure that they are not omitted.

<details>
  <summary>
    <b>Parameter details</b>
  </summary>

- `--gpus all`: Allows the container to access all GPUs on the host.

- `--network host`: The container shares the host's network. This is required for enabling RDMA.

- `--name lmdeploy`: Specifies the container name as `lmdeploy`.

- `--ipc host`: The container shares shared memory with the host, which accelerates inter-process communication. There's no need to set shm-size separately.

- `--ulimit memlock=-1`: The container can use unlimited memory, preventing slowdowns in inference performance due to insufficient memory.

- `-v $MODEL_PATH:$CONTAINER_MODEL_PATH`: Mounts the dsv3 model located at `MODEL_PATH` on the host into the container at the path `CONTAINER_MODEL_PATH`.

- `$device_args`: Mounts the detected RDMA devices into the container for RDMA-accelerated communication. If RDMA is not used, this parameter can be removed.

</details>

Command to re-enter the container

```bash
docker exec -it lmdeploy /bin/bash
```

## 2. Build a multi-node cluster using Ray

> :warning: The following operations, including starting a multi-node cluster, launching the API server, and performance testing, are all assumed to be performed within the Docker container.
> We will build a Ray cluster consisting of docker containers, therefore commands executed on the host machine terminal won't be able to access this cluster.

LMdeploy utilizes Ray for multi-node cluster construction, as described in [LMDeploy multi-node deployment](https://lmdeploy.readthedocs.io/en/latest/advance/pytorch_multinodes.html).

In the following steps, we will build a Ray cluster with two nodes for illustration.

### 2.1 Start multi-node Ray cluster

Start the ray cluster on the head node. The default port in Ray is 6379, you may modify it according to your needs.

```bash
ray start --head --port=6379
```

Start on the slave nodes to join in the ray cluster, suppose the head node ip is `10.130.8.143` and port is `6379` (change it to your own):

```bash
ray start --address=10.130.8.143:6379
```

### 2.2 Check cluster status

Use the following commands to check the ray cluster status on both head and slave nodes. You should be able to see the ray cluster status of multiple nodes information.

```bash
ray status
```

## 3. Launch DeepSeekV3 API service

Use the following commands to launch the LMDeploy DeepSeekV3 API service. We currently support TP16 deployment.

```bash
MODEL_PATH='xxx'

lmdeploy serve api_server \
    $MODEL_PATH \
    --backend pytorch \
    --tp 16
```

## 4. Benchmarking the inference performance

To benchmark LMDeploy DeepSeekV3 inference performance, you may refer to the following scripts and modify the parameters according to your needs.

```bash
#!/bin/bash

num_prompts=10000
backend="lmdeploy"
dataset_name="random"
dataset_path="./benchmark/ShareGPT_V3_unfiltered_cleaned_split.json"

echo ">>> tp: ${tp}, num_prompts: ${num_prompts}, dataset: ${dataset_name}"

for in_len in 2048
do
    echo "input len: ${in_len}"

    for out_len in 1024 2048 4096 8192
    do
        echo "output len: ${out_len}"

        python3 benchmark/profile_restful_api.py \
            --backend ${backend} \
            --dataset-name ${dataset_name} \
            --dataset-path ${dataset_path} \
            --num-prompts ${num_prompts} \
            --random-input-len ${in_len} \
            --random-output-len ${out_len} \
            --output-file "./benchmark/res/api_n${num_prompts}_in${in_len}_out${out_len}_dsv3.csv"
    done

done

```

## 5. LMDeploy benchmarking results

We have benchmarked the DeepSeekV3 inference performance in the following table for your reference.

| max bsz | prompt no | input-len | output-len | input token throughput (tok/s) | output token throughput (tok/s) | RPS  |
| ------- | :-------: | :-------: | :--------: | :----------------------------: | :-----------------------------: | :--: |
| 1024    |   10000   |   2048    |    1024    |            3489.50             |             1743.56             | 3.4  |
| 1024    |   10000   |   2048    |    2048    |            1665.07             |             1682.41             | 1.62 |
| 1024    |   10000   |   2048    |    4096    |                                |                                 |      |
| 1024    |   10000   |   2048    |    8192    |                                |                                 |      |
| 128     |   2000    |   2048    |   16384    |             76.78              |             600.07              | 0.07 |
| 128     |   2000    |   2048    |   32768    |             17.75              |             281.89              | 0.02 |

For output lengths of 16k and 32k, we decrease the total prompt numbers to shorten the execution time.

## 6. Optimize inference efficiency

- RDMA is required for fast inter-node communications. Make sure you have RDMA enabled, otherwise performance will be sub-optimal.
