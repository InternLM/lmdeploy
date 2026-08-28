# 使用 checkpoint-engine 更新 PyTorch 权重

LMDeploy PyTorch Engine 可以通过 CUDA IPC 接收
[MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)
提供的模型权重。Broadcast 和 P2P 共用同一条 LMDeploy 接收链路，区别在于
ParameterServer 如何把权重送入最终导出的 GPU buffer。

## 支持范围

| 项目     | 支持情况                                        |
| -------- | ----------------------------------------------- |
| Backend  | PyTorch                                         |
| 设备     | CUDA                                            |
| 并行     | TP、DP、EP；目标必须包含完整 TP 和 EP group     |
| 权重     | FP16/BF16，以及 PyTorch 已支持的 FP8/MoE loader |
| 生命周期 | 空权重初始化、显式在线 sleep/wakeup             |

当前集成不支持 Turbomind、非 CUDA 设备、部分 TP/EP group 更新、EPLB 和
MemDecode；AWQ/GPTQ 尚未纳入验收范围。

安装已验证的 checkpoint-engine 版本。仅使用 Broadcast 时安装基础包即可；
`p2p` extra 会同时安装 Mooncake Transfer Engine：

```bash
pip install 'checkpoint-engine==0.4.2'          # Broadcast
pip install 'checkpoint-engine[p2p]==0.4.2'
```

P2P 还要求系统提供 `libnuma.so.1`（Ubuntu 的 `libnuma1` 包）以及可用的
RDMA 环境。如果 Mooncake 发现的 HCA 数量与可见 GPU 数量不兼容，需要显式
选择 HCA：

```bash
export PS_P2P_STORE_RDMA_DEVICES=mlx5_0,mlx5_1
```

安装 P2P extra 后，checkpoint-engine 即使执行 Broadcast 也会初始化 P2PStore，
因此该环境下 Broadcast 也可能需要设置 HCA。向 LMDeploy 提交 CUDA IPC buffer
的 ParameterServer 进程必须与对应 LMDeploy worker 位于同一块物理 GPU，传输前
需要校验 CUDA UUID。
该共卡要求针对最终导出 CUDA IPC buffer 的 inference-side ParameterServer。
P2P source ParameterServer 可以保留在远端训练节点；inference-side joining
ParameterServer 先通过 Mooncake/RDMA 从 source 拉取，再在本地导出给 LMDeploy。

## 启动空权重 PyTorch 服务

显式选择 Ray 并使用 `--empty-init`。LMDeploy 会构建模型结构，但不加载权重、
不分配 KV cache，也不执行 warmup：

```bash
lmdeploy serve api_server /path/to/model-config-and-tokenizer \
  --backend pytorch \
  --distributed-executor-backend ray \
  --empty-init \
  --tp 2 \
  --api-keys YOUR_API_KEY
```

DP 场景启动 API server 前还需要为所有 DP worker 配置共享 rendezvous：

```bash
export LMDEPLOY_DP_MASTER_ADDR=10.0.0.10
export LMDEPLOY_DP_MASTER_PORT=29500
```

EP 必须与 Ray executor 配合。以 Qwen3.5-35B-A3B 的 DP4/EP4 为例，四个
API endpoint 分别拥有一个本地 worker，但四个 global rank 共同组成一个 EP
group：

```bash
lmdeploy serve api_server /path/to/Qwen3.5-35B-A3B \
  --backend pytorch \
  --distributed-executor-backend ray \
  --empty-init \
  --tp 1 --dp 4 --ep 4 \
  --api-keys YOUR_API_KEY
```

此拓扑中的 Broadcast 和 P2P 都必须一起更新四个 endpoint。只更新一个 DP
endpoint 会遗漏其他 rank 持有的 expert，驱动必须在传输前拒绝这种请求。

此时服务处于 healthy-but-sleeping 状态，在权重和 KV cache 都就绪前会拒绝
推理请求。

启动 ParameterServer 传输前检查接收端：

```bash
curl -H 'Authorization: Bearer YOUR_API_KEY' \
  http://127.0.0.1:23333/update_weights_from_ipc
```

就绪响应会按本地 worker rank 返回 CUDA UUID：

```json
{
  "ready": true,
  "message": "checkpoint-engine IPC receiver is ready.",
  "backend": "pytorch",
  "device_type": "cuda",
  "checkpoint_engine_version": "0.4.2",
  "is_sleeping": true,
  "sleeping_tags": ["kv_cache", "weights"],
  "device_uuids": ["GPU-..."],
  "worker_ranks": [0],
  "world_size": 4,
  "tp": 1,
  "dp": 4,
  "dp_rank": 0,
  "ep": 4
}
```

多 endpoint 场景必须按 `dp_rank` 排列 URL。驱动需要同时校验
`worker_ranks`、`world_size`、`tp`、`dp`、`dp_rank` 和 `ep`，不能只按 UUID 数量
推断并行拓扑。

## ParameterServer 回调

每个 LMDeploy model worker 对应一个 ParameterServer 进程。每个完整推理并行
group 的源 rank 将该 group 的 UUID/ZMQ 映射发送给 LMDeploy：

```python
import httpx
import os

rank = int(os.environ['RANK'])
inference_parallel_size = 2
source_rank = rank // inference_parallel_size * inference_parallel_size


def request_lmdeploy(socket_paths):
    if rank != source_rank:
        return
    handles = dict(socket_paths[source_rank:source_rank + inference_parallel_size])
    response = httpx.post(
        'http://127.0.0.1:23333/update_weights_from_ipc',
        headers={'Authorization': 'Bearer YOUR_API_KEY'},
        json={'zmq_handles': handles},
        timeout=600,
    )
    response.raise_for_status()
```

调用 `update` 前，按照 checkpoint-engine 的 `ParameterServer` API 注册并
gather checkpoint。多 ParameterServer rank 之间必须按照排序后的 checkpoint
文件名或 tensor name 进行确定性分片。

### Broadcast

不设置 `ranks`。checkpoint-engine 将注册的 checkpoint Broadcast 到全部
ParameterServer rank，再由各 rank 向同 GPU 的 LMDeploy worker 导出 buffer：

```python
ps.update('checkpoint-name', request_lmdeploy, ranks=None)
```

更新成功只代表权重已就绪，推理仍然保持阻塞。需要显式分配并 warmup KV cache：

```bash
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=kv_cache'
```

### P2P

通过 `ranks` 传入完整目标 group，checkpoint-engine 会选择 Mooncake P2P
路径：

```python
# 更新一个完整的 TP=2 副本。
ps.update('checkpoint-name', request_lmdeploy, ranks=[0, 1])
```

新 ParameterServer 加入时，需要保持源 ParameterServer 存活，使用
`load_metas` 加载源端序列化 metadata，并使用新 ParameterServer 的本地 CUDA
UUID 拓扑调用 LMDeploy。源 pinned memory 注册和 Mooncake endpoint 必须保持到
传输结束。

禁止只传入 LMDeploy TP 或 EP group 的一部分。DP/EP 模式下必须直接并发调用
每个相关 DP API server；sleep/wakeup 和 EP 推理可能进入跨 rank collective，
顺序 fan-out 会死锁。
LMDeploy proxy 不会自动 fan-out 权重管理请求。

例如 DP4/EP4 的 P2P 更新必须传入完整 group：

```python
ps.update('checkpoint-name', request_lmdeploy, ranks=[0, 1, 2, 3])
```

## 在线更新

该集成不会自动 drain 请求。必须显式 sleep 目标实例、恢复或重建权重空间、
更新权重，最后再唤醒 KV cache：

```bash
# 1. 阻止新请求、排空运行中任务并释放运行时显存。
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/sleep?level=2'

# 2. 重建空的 CUDA 权重空间。
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=weights'

# 3. 检查 GET /update_weights_from_ipc，然后运行 ps.update(...)。

# 4. 重建 KV cache 并恢复推理。
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=kv_cache'
```

Level 1 保留旧权重的 CPU 副本；Level 2 丢弃旧权重，并在更新前重建空模型。

## 失败恢复

IPC 更新不具备事务性。传输开始后任一 worker 失败，LMDeploy 都会报告权重可能
只更新了一部分，并继续保持 sleeping。此时不要调用 `wakeup(kv_cache)`；应停止
并重建全部受影响的 DP 副本，再重新执行 readiness 和完整更新。

排查失败或阻塞时，依次确认：

- 所有 endpoint 都返回 `ready=true`；
- ParameterServer 与 LMDeploy 的 UUID group 按 rank 完全一致；
- 所有目标 rank 都组成完整 TP 和 EP group；
- P2P 源 ParameterServer 和 metadata 仍然有效；
- readiness 和 update 请求都携带 API key；
- P2P 环境存在可用 Mooncake/RDMA 设备。

在 `/dev/shm` 下注册 safetensors 时，checkpoint-engine 的 in-place pin 模式
可能删除原文件。除非明确接受该破坏性行为，否则应保持禁用。
