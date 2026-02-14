# DeepSeekV3 Benchmarking

## Benchmark results

______________________________________________________________________

### v0.7.1 + `9528a74`

2 nodes, TP16

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| 1024    |   10000   |   2048    |    1024    |             1744.75              |              871.78               |
| 1024    |   10000   |   2048    |    2048    |              832.54              |              841.21               |
| 1024    |   10000   |   2048    |    4096    |              362.51              |              727.56               |
| 1024    |   10000   |   2048    |    8192    |              126.59              |              504.90               |
| Default |   2000    |   2048    |   16384    |              38.39               |              300.04               |
| Default |   2000    |   2048    |   32768    |               8.88               |              140.95               |

### v0.7.2 + `1e77ed2`

2 nodes, TP16

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| 1024    |   10000   |   2048    |    1024    |             1812.15              |              905.46               |
| 1024    |   10000   |   2048    |    2048    |              876.88              |              886.01               |
| 1024    |   10000   |   2048    |    4096    |              382.31              |              767.29               |
| 1024    |   10000   |   2048    |    8192    |              140.66              |              561.04               |
| 1024    |   2000    |   2048    |   16384    |              49.46               |              386.53               |
| 1024    |   2000    |   2048    |   32768    |              14.88               |              236.37               |

For output lengths of 16k and 32k, we decrease the number of total prompts to shorten the execution time.

### v0.7.2 + `f24497f`

4 nodes, DP32 + EP32

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| Default |   2000    |   2048    |    1024    |             2066.14              |              1016.13              |
| Default |   2000    |   2048    |    2048    |             1516.38              |              1504.41              |
| Default |   2000    |   2048    |    4096    |              916.75              |              1799.87              |
| Default |   2000    |   2048    |    8192    |              493.12              |              2002.58              |
| Default |   2000    |   2048    |   16384    |              248.39              |              1941.23              |
| Default |   2000    |   2048    |   32768    |              109.50              |              1739.20              |

### v0.7.3 + `665f54f`

2 nodes, TP16

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| 1024    |   10000   |   2048    |    2048    |              964.64              |              974.68               |

## User guide (2 nodes)

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

Start on the slave nodes to join the ray cluster, change the IP and port to your own:

```bash
ray start --address=${head_node_ip}:6379
```

Use the following commands to check the ray cluster status on both head and slave nodes. You should be able to see the ray cluster status of multiple nodes.

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

echo ">>> num_prompts: ${num_prompts}, dataset: ${dataset_name}"

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
            --random-output-len ${out_len}
    done

done

```

## User guide (4 nodes)

______________________________________________________________________

### Installation

To achieve the best inference efficiency when depolying DeepSeekV3, we recommend that users adopt our `openmmlab/lmdeploy:latest-cu12-hopper` for Hopper GPU. In this docker image, we have pre-installed Hopper-specific third-party libraries such as [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), and [DeepEP](https://github.com/deepseek-ai/DeepEP) to speed up the DeepSeekV3 inference.

Use the following commands to create containers on each of the 4 nodes.

```bash
docker run -it \
    --gpus all \
    --network host \
    --ipc host \
    --name lmdeploy \
    --privileged \
    -v "/path/to/the/huggingface/home/in/this/node":"root/.cache/huggingface" \
    openmmlab/lmdeploy:latest-cu12-hopper
```

### Start proxy server

With 4 nodes, DP32 EP32 settings, LMDeploy utilizes a proxy server to handle the request distribution. Therefore, we need to start a proxy server on the master node as follows.

```bash
lmdeploy serve proxy --server-name ${master_node_ip} --server-port ${proxy_server_port} --strategy "min_expected_latency"
```

### Launch service

Execute the following commands on 4 nodes.

```bash
# node0
torchrun --nnodes=4 --nproc_per_node 8 --node_rank=0 --master_addr=${master_node_ip} --master_port=29500 dpep_torchrun_serve.py --proxy_url  http://${master_node_ip}:${proxy_server_port}

# node1
torchrun --nnodes=4 --nproc_per_node 8 --node_rank=1 --master_addr=${master_node_ip} --master_port=29500 dpep_torchrun_serve.py --proxy_url  http://${master_node_ip}:${proxy_server_port}

# node2
torchrun --nnodes=4 --nproc_per_node 8 --node_rank=2 --master_addr=${master_node_ip} --master_port=29500 dpep_torchrun_serve.py --proxy_url  http://${master_node_ip}:${proxy_server_port}

# node3
torchrun --nnodes=4 --nproc_per_node 8 --node_rank=3 --master_addr=${master_node_ip} --master_port=29500 dpep_torchrun_serve.py --proxy_url  http://${master_node_ip}:${proxy_server_port}
```

To facilitate the launch, we provide a script named `dpep_torchrun_serve.py`

<details>
  <summary>
    <b>dpep_torchrun_serve.py</b>
  </summary>

```python
import os
import fire
import socket
from typing import List, Literal


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def main(
    model_path: str = "deepseek-ai/DeepSeek-V3",
    tp: int = 1,
    dp: int = 32,
    ep: int = 32,
    proxy_url: str = "http://${master_node_ip}:${proxy_server_port}",
    port: int = 23333,
    backend: str = "pytorch",
):

    # get distributed env parameters
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    global_rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # get current node api ip and port
    local_ip = get_host_ip()
    if isinstance(port, List):
        assert len(port) == world_size
        port = port[global_rank]
    else:
        port += global_rank * 10

    print(
        f"tp: {tp}, dp: {dp}, ep: {ep}, world_size: {world_size}, "
        f"global_rank: {global_rank}, local_rank: {local_rank}, "
        f"ip: {local_ip}, port: {port}"
    )

    # set lmdeploy DP distributed env variables
    os.environ['LMDEPLOY_DP_MASTER_ADDR'] = ${master_node_ip}
    os.environ['LMDEPLOY_DP_MASTER_PORT'] = str(29555)

    # build command with node-aware GPU assignment
    dp_rank = global_rank

    command = (
        f"CUDA_VISIBLE_DEVICES={local_rank} lmdeploy serve api_server {model_path} "
        f"--cache-max-entry-count 0.7 "
        f"--max-prefill-token-num 1000 "
        f"--server-name {local_ip} --server-port {port} "
        f"--tp {tp} --dp {dp} --ep {ep} --dp-rank {dp_rank} "
        f"--proxy-url {proxy_url} --backend {backend}"
    )

    print(f"Running command: {command}")
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
```

</details>

### Wakeup and warmup engine

After launching the API service, we need one more step to warm up the engine.
Modify the node0 ~ node3 IP addresses in the following script and execute it to warm up the engine.

Once you see `Warm up finished, feel free to go!`, use the DeepSeekV3 API service with `http://${master_node_ip}:${proxy_server_port}` as usual.

<details>
  <summary>
    <b>warmup_engine.py</b>
  </summary>

```python
import asyncio
from openai import OpenAI


async def wake_up_node(dp_rank):
    text_prompt = "The quick brown fox jumps over the lazy dog."

    messages = [
        {"role": "user", "content": [{"type": "text", "text": text_prompt}]}
    ]

    base_port = 23333 + (dp_rank * 10)
    # node0
    if 0 <= dp_rank < 8:
        node_ip = ${node0_ip}
    # node1
    elif 8 <= dp_rank < 16:
        node_ip = ${node1_ip}
    # node2
    elif 16 <= dp_rank < 24:
        node_ip = ${node2_ip}
    # node3
    elif 24 <= dp_rank < 32:
        node_ip = ${node3_ip}

    base_url = f"http://{node_ip}:{base_port}/v1"
    print(f"wake up => {base_url}")

    # initialize the OpenAI client
    client = OpenAI(api_key="YOUR_API_KEY", base_url=base_url)

    try:
        # await the coroutine returned by asyncio.to_thread
        model_list = await asyncio.to_thread(client.models.list)
        model_name = model_list.data[0].id

        # await the coroutine for chat completion
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            max_tokens=20,
        )
        print(f"Response from {base_url}: {response}\n")
    except Exception as e:
        print(f"Error waking up {base_url}: {e}")


async def wake_up(dp_size):
    # create tasks for all ranks
    tasks = [wake_up_node(dp_rank) for dp_rank in range(dp_size)]
    await asyncio.gather(*tasks)


# run the asynchronous wake-up function
if __name__ == "__main__":
    dp_size = 32
    asyncio.run(wake_up(dp_size))
    print(">" * 50)
    print("Warm up finished, feel free to go!")
    print("<" * 50)
```

</details>
