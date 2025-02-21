# PyTorchEngine Multi-Node Deployment Guide

To support larger-scale model deployment requirements, PyTorchEngine provides multi-node deployment support. Below are the detailed steps for deploying a `tp=16` model across two 8-card nodes.

## 1. Create Docker Containers (Optional)

To ensure consistency across the cluster environment, it is recommended to use Docker to set up the cluster. Create containers on each node as follows:

```bash
docker run -it \
    --network host \
    -v $MODEL_PATH:$CONTAINER_MODEL_PATH \
    openmmlab/lmdeploy:latest
```

> [!IMPORTANT]
> Ensure that the model is placed in the same directory on all node containers.

## 2. Set Up the Cluster Using Ray

### 2.1 Start the Head Node

Select one node as the **head node** and run the following command in its container:

```bash
ray start --head --port=$DRIVER_PORT
```

### 2.2 Join the Cluster

On the other nodes, use the following command in their containers to join the cluster created by the head node:

```bash
ray start --address=$DRIVER_NODE_ADDR:$DRIVER_PORT
```

run `ray status` on head node to check the cluster.

> [!IMPORTANT]
> Ensure that `DRIVER_NODE_ADDR` is the address of the head node and `DRIVER_PORT` matches the port number used during the head node initialization.

## 3. Use LMDeploy Interfaces

In the head node's container, you can use all functionalities of PyTorchEngine as usual.

### 3.1 Start the Server

```bash
lmdeploy serve api_server \
    $CONTAINER_MODEL_PATH \
    --backend pytorch \
    --tp 16
```

### 3.2 Use the Pipeline

```python
from lmdeploy import pipeline, PytorchEngineConfig

if __name__ == '__main__':
    model_path = '/path/to/model'
    backend_config = PytorchEngineConfig(tp=16)
    with pipeline(model_path, backend_config=backend_config) as pipe:
        outputs = pipe('Hakuna Matata')
```

> [!NOTE]
> PyTorchEngine will automatically choose the appropriate launch method (single-node/multi-node) based on the `tp` parameter and the number of devices available in the cluster. If you want to enforce the use of the Ray cluster, you can configure `distributed_executor_backend='ray'` in `PytorchEngineConfig` or use the environment variable `LMDEPLOY_EXECUTOR_BACKEND=ray`.

---

By following the steps above, you can successfully deploy PyTorchEngine in a multi-node environment and leverage the Ray cluster for distributed computing.

> [!WARNING]
> To achieve better performance, we recommend users to configure a higher-quality network environment (such as using [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand)) to improve engine efficiency.
