# PyTorchEngine 多节点部署指南

为了支持更大规模的模型部署需求，PyTorchEngine 提供了多节点部署的支持。以下是如何在两个8卡节点上部署 tp=16 模型的详细步骤。

## 1. 创建 Docker 容器（可选）

为了确保集群环境的一致性，建议使用 Docker 搭建集群。在每个节点上创建容器：

```bash
docker run -it \
    --network host \
    -v $MODEL_PATH:$CONTAINER_MODEL_PATH \
    openmmlab/lmdeploy:latest
```

> \[!IMPORTANT\]
> 请确保将模型放置在各个节点容器的相同目录中。

## 2. 使用 ray 搭建集群

### 2.1 启动主节点

选择其中一个节点做为`主节点`，并在该节点的容器中运行以下命令：

```bash
ray start --head --port=$DRIVER_PORT
```

### 2.2 加入集群

在其他节点的容器中，使用以下命令加入主节点所在的集群：

```bash
ray start --address=$DRIVER_NODE_ADDR:$DRIVER_PORT
```

完成后可以在主节点使用 `ray status` 查看集群状态，确保所有节点都被成功加入集群。

> \[!IMPORTANT\]
> 请确保 `DRIVER_NODE_ADDR` 为主节点的地址，`DRIVER_PORT` 与主节点初始化时使用的端口号一致。

## 3. 使用 LMDeploy 接口

在主节点的容器中，您可以正常使用 PyTorchEngine 的所有功能。

### 3.1 启动服务 API

```bash
lmdeploy serve api_server \
    $CONTAINER_MODEL_PATH \
    --backend pytorch \
    --tp 16
```

### 3.2 使用 pipeline 接口

```python
from lmdeploy import pipeline, PytorchEngineConfig

if __name__ == '__main__':
    model_path = '/path/to/model'
    backend_config = PytorchEngineConfig(tp=16)
    with pipeline(model_path, backend_config=backend_config) as pipe:
        outputs = pipe('Hakuna Matata')
```

> \[!NOTE\]
> PytorchEngine 会根据 tp 数以及集群上的设备数量自动选择合适的启动方式（单机/多机）。如果希望强制使用 ray 集群，可以配置 `PytorchEngineConfig` 中的 `distributed_executor_backend='ray'` 或使用环境变量 `LMDEPLOY_EXECUTOR_BACKEND=ray`。

通过以上步骤，您可以成功在多节点环境中部署 PyTorchEngine，并利用 Ray 集群进行分布式计算。

> \[!WARNING\]
> 为了能够得到更好的性能，我们建议用户配置更好的网络环境（比如使用 [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand)）以提高引擎运行效率
