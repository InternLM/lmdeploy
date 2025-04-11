# DeepSeekV3 Benchmarking

## 性能测试结果

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

### v0.7.2 + `1e77ed2`

| max bsz | prompt no | input-len | output-len | input token throughput (tok/s) | output token throughput (tok/s) |
| ------- | :-------: | :-------: | :--------: | :----------------------------: | :-----------------------------: |
| 1024    |   10000   |   2048    |    1024    |            3624.30             |             1810.91             |
| 1024    |   10000   |   2048    |    2048    |            1753.75             |             1772.01             |
| 1024    |   10000   |   2048    |    4096    |             764.61             |             1534.58             |
| 1024    |   10000   |   2048    |    8192    |             281.32             |             1122.08             |
| 1024    |   2000    |   2048    |   16384    |             98.92              |             773.06              |
| 1024    |   2000    |   2048    |   32768    |             29.76              |             472.74              |

对于输出长度为 16k 和 32k 的情况，我们减少了prompt no以缩短实验时间。

### v0.7.2 + `f24497f`

| max bsz | prompt no | input-len | output-len | input token throughput (tok/s) | output token throughput (tok/s) |
| ------- | :-------: | :-------: | :--------: | :----------------------------: | :-----------------------------: |
| 128     |   2000    |   2048    |    1024    |            8264.56             |             4064.52             |
| 128     |   2000    |   2048    |    2048    |            6065.50             |             6017.63             |
| 128     |   2000    |   2048    |    4096    |            3666.99             |             7199.48             |
| 128     |   2000    |   2048    |    8192    |            1972.46             |             8010.31             |
| 128     |   2000    |   2048    |   16384    |             993.55             |             7764.91             |
| 128     |   2000    |   2048    |   32768    |             438.00             |             6956.78             |

注意： 本性能数据是在使用 4 个节点（DP32 + EP32）的情况下测量的，而之前的结果是在使用 2 个节点（TP16）的情况下记录的。

## 用户指南

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
