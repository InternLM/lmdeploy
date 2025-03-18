# DeepSeekV3 Benchmarking

## Benchmark results

______________________________________________________________________

### v0.7.1 + `9528a74`

| max bsz | prompt no | input-len | output-len | input token throughput (tok/s) | output token throughput (tok/s) |
| ------- | :-------: | :-------: | :--------: | :----------------------------: | :-----------------------------: |
| 1024    |   10000   |   2048    |    1024    |            3489.50             |             1743.56             |
| 1024    |   10000   |   2048    |    2048    |            1665.07             |             1682.41             |
| 1024    |   10000   |   2048    |    4096    |             725.01             |             1455.12             |
| 1024    |   10000   |   2048    |    8192    |             253.17             |             1009.80             |
| 128     |   2000    |   2048    |   16384    |             76.78              |             600.07              |
| 128     |   2000    |   2048    |   32768    |             17.75              |             281.89              |

For output lengths of 16k and 32k, we decrease the total prompt numbers to shorten the execution time.

## User guide

______________________________________________________________________

### Installation

In this document, we will provide step-by-step guidance on how to set up DeepSeekV3 inference with LMDeploy on a multi-node cluster.

We highly recommend that users adopt our official docker image to avoid potential errors caused by environmental differences. Execute the following commands on both head and slave nodes to create docker containers.

```bash
docker run -it \
    --gpus all \
    --network host \
    --ipc host \
    --name lmdeploy \
    --privileged \
    -v "/path/to/the/huggingface/home/in/this/node":"root/.cache/huggingface" \
    openmmlab/lmdeploy:latest-cu12
```

`--privileged` is required for enabling RDMA.

### Build a multi-node cluster using Ray

> :warning: The following operations are all assumed to be performed within the Docker container.
> We will build a Ray cluster consisting of docker containers, therefore commands executed on the host machine terminal won't be able to access this cluster.

LMdeploy utilizes Ray for multi-node cluster construction. In the following steps, we will build a Ray cluster with two nodes for illustration.

Start the ray cluster on the head node. The default port in Ray is 6379 (change it to your own).

```bash
ray start --head --port=6379
```

Start on the slave nodes to join in the ray cluster, suppose the head node ip is `xxx` and port is `6379` (change it to your own):

```bash
ray start --address=xxx:6379
```

Use the following commands to check the ray cluster status on both head and slave nodes. You should be able to see the ray cluster status of multiple nodes information.

```bash
ray status
```

### Launch service

Use the following commands to launch the LMDeploy DeepSeekV3 API service. We currently support TP16 deployment.

```bash
lmdeploy serve api_server deepseek-ai/DeepSeek-V3 --backend pytorch --tp 16
```

### Benchmarking

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
