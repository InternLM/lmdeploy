# 权重更新

LMDeploy支持在线权重更新，方便RL训练等场景下的使用。以下是权重更新的步骤：

checkpoint-engine Broadcast 和 Mooncake P2P 的使用方法请参考
[使用 checkpoint-engine 更新 PyTorch 权重](./checkpoint_engine.md)。

`POST /update_weights` **默认**拒绝 pickle 载荷。请使用
`load_format="safetensors"`（推荐）或结构化 tensor dict。
默认 `api_server` 绑定下，未认证的 HTTP pickle 反序列化属于远程代码执行，因此保持关闭，除非显式开启。

XTuner 同机 CUDA IPC 仍通过 HTTP `/update_weights` 发送 `serialize_state_dict()` /
`FlattenedTensorBucket` 控制消息（IPC handle、event handle、`FlattenedTensorMetadata`）。
张量本身留在同机 GPU 内存中，只有控制消息走 HTTP。该路径（包括仅发送 metadata 的 buffer 复用，以及空的 `finished=true` 收尾请求）需要在**服务端**进程设置
`LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1` 才能恢复。Ray worker 会通过 `get_all_envs()` 继承该变量。不要在不受信任的公网入口上开启该选项。

## 步骤 1: 启动服务

For pytorch backend you have to add `--distributed-executor-backend ray`.

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --distributed-executor-backend ray # for pytorch backend
```

## 步骤 2: 卸载权重和KV缓存

在权重更新前，需要调用API卸载权重和KV缓存，使推理引擎处于可更新状态：

```python
from lmdeploy.utils import serialize_named_tensors_safetensors
import requests

BASE_URL = 'http://0.0.0.0:23333'
api_key = 'sk-xxx'

headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

# offloads weights and kv cache with level=2
response = requests.post(f"{BASE_URL}/sleep", headers=headers, params=dict(tags=['weights', 'kv_cache'], level=2))
assert response.status_code == 200, response.status_code

# wake up weights, the server is ready for update weights
response = requests.post(f"{BASE_URL}/wakeup", headers=headers, params=dict(tags=['weights']))
assert response.status_code == 200, response.status_code
```

## 步骤 3: 更新权重

将模型权重切分后调用`update_weights`API进行更新。

```python
segmented_state_dict: List[Dict[str, torch.Tensor]] = ...
num_segment = len(segmented_state_dict)
for seg_idx in range(num_segment):
    serialized_data = serialize_named_tensors_safetensors(segmented_state_dict[seg_idx])
    data = dict(serialized_named_tensors=serialized_data, load_format='safetensors', finished=seg_idx == num_segment-1)
    response = requests.post(f"{BASE_URL}/update_weights", headers=headers, json=data)
    assert response.status_code == 200, f"response.status_code = {response.status_code}"

```

PyTorch 还可以通过 `POST /update_weights_from_distributed`（NCCL）和
`POST /update_weights_from_ipc`（checkpoint-engine）接收权重，这两条路径不会对 HTTP
body 做 pickle 反序列化。

**注意**: XTuner 通过 HTTP `/update_weights` 做 flattened-bucket CUDA IPC 时，需要在服务端设置
`LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1`：

```python
from lmdeploy.utils import serialize_state_dict, FlattenedTensorBucket, FlattenedTensorMetadata

segmented_state_dict: List[Dict[str, torch.Tensor]] = ...
num_segment = len(segmented_state_dict)
for seg_idx in range(num_segment):
    named_tensors = list(segmented_state_dict[seg_idx].items())
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    metadata = bucket.get_metadata()
    flattened_tensor_data = dict(flattened_tensor=bucket.get_flattened_tensor(), metadata=metadata)
    serialized_data = serialize_state_dict(flattened_tensor_data)
    data = dict(serialized_named_tensors=serialized_data, finished=seg_idx == num_segment-1, load_format='flattened_bucket')
    response = requests.post(f"{BASE_URL}/update_weights", headers=headers, json=data)
    assert response.status_code == 200, f"response.status_code = {response.status_code}"
```

## 步骤 4: 唤醒引擎

权重更新后，调用API构建KV缓存，唤醒引擎，重新提供推理服务。

```python
response = requests.post(f"{BASE_URL}/wakeup", headers=headers, params=dict(tags=['kv_cache']))
assert response.status_code == 200, response.status_code
```
