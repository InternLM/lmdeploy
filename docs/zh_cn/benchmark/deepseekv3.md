# DeepSeekV3 基准测试

## 性能测试结果

______________________________________________________________________

### v0.7.1 + `9528a74`

2 节点, TP16

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| 1024    |   10000   |   2048    |    1024    |             1744.75              |              871.78               |
| 1024    |   10000   |   2048    |    2048    |              832.54              |              841.21               |
| 1024    |   10000   |   2048    |    4096    |              362.51              |              727.56               |
| 1024    |   10000   |   2048    |    8192    |              126.59              |              504.90               |
| Default |   2000    |   2048    |   16384    |              38.39               |              300.04               |
| Default |   2000    |   2048    |   32768    |               8.88               |              140.95               |

### v0.7.2 + `1e77ed2`

2 节点, TP16

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

4 节点, DP32 + EP32

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| Default |   2000    |   2048    |    1024    |             2066.14              |              1016.13              |
| Default |   2000    |   2048    |    2048    |             1516.38              |              1504.41              |
| Default |   2000    |   2048    |    4096    |              916.75              |              1799.87              |
| Default |   2000    |   2048    |    8192    |              493.12              |              2002.58              |
| Default |   2000    |   2048    |   16384    |              248.39              |              1941.23              |
| Default |   2000    |   2048    |   32768    |              109.50              |              1739.20              |

### v0.7.3 + `665f54f`

2 节点, TP16

| max bsz | prompt no | input-len | output-len | per node input token thr (tok/s) | per node output token thr (tok/s) |
| ------- | :-------: | :-------: | :--------: | :------------------------------: | :-------------------------------: |
| 1024    |   10000   |   2048    |    2048    |              964.64              |              974.68               |

## 用户指南（两节点）

______________________________________________________________________

### 安装

在本文档中，我们将提供在多节点集群上使用 LMDeploy 部署 DeepSeekV3 的详细教程。

我们强烈建议用户采用官方 Docker 镜像，以避免因环境差异而导致的潜在问题。请在主节点和从节点上执行以下命令来创建 Docker 容器。

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

其中 `--privileged` 是开启 RDMA 必需的参数。

### 用Ray构建多节点集群

> :warning: 以下所有操作均默认在 Docker 容器内执行。
> 我们将构建一个由 Docker 容器组成的 Ray 集群，因此在宿主机终端上执行的命令将无法访问该集群。

LMdeploy 使用 Ray 来构建多节点集群。在接下来的步骤中，我们将以两个节点为例构建一个 Ray 集群。

在主节点上启动 Ray 集群。Ray 的默认端口是 6379（请按需修改）。

```bash
ray start --head --port=6379
```

在从节点上启动并加入 Ray 集群，请按需修改主节点 ip 和 port：

```bash
ray start --address=${head_node_ip}:6379
```

使用以下命令在主节点和从节点上检查 Ray 集群状态。您应该能够看到包含多节点信息的 Ray 集群状态。

```bash
ray status
```

### 启动服务

使用以下命令启动 LMDeploy DeepSeekV3 API 服务。我们目前支持 TP16 部署。

```bash
lmdeploy serve api_server deepseek-ai/DeepSeek-V3 --backend pytorch --tp 16
```

### 性能测试

要对 LMDeploy DeepSeekV3 的推理性能进行基准测试，您可以参考以下脚本，并根据需要修改参数。

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

## 用户指南（四节点）

______________________________________________________________________

### 安装

为了在部署 DeepSeekV3 时实现最佳的推理效率，我们建议用户采用我们的 openmmlab/lmdeploy:latest-cu12-hopper 镜像以适配 Hopper GPU。在此 Docker 镜像中，我们预先安装了针对 Hopper 架构优化的第三方库，例如 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 和 [DeepEP](https://github.com/deepseek-ai/DeepEP) ，以加速 DeepSeekV3 的推理过程。

使用以下命令在每个节点（共 4 个节点）上创建容器。

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

### 启动服务

在 4 个节点上执行以下命令。

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

为便于启动，我们提供了一个名为 `dpep_torchrun_serve.py` 的脚本。

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

### 引擎唤醒与预热

在启动 API 服务后，我们还需要一个步骤才能准备就绪。修改以下脚本中的 node0 ~ node3 的 IP 地址，并执行以预热引擎。

当看到 `Warm up finished, feel free to go!` 信息，即可通过 `http://${master_node_ip}:${proxy_server_port}` 正常使用 DeepSeekV3 API 服务。

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
