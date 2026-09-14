# 权重更新

LMDeploy支持在线权重更新，方便RL训练等场景下的使用。以下是权重更新的步骤：

checkpoint-engine Broadcast 和 Mooncake P2P 的使用方法请参考
[使用 checkpoint-engine 更新 PyTorch 权重](./checkpoint_engine.md)。

`POST /update_weights` 不再接受 pickle 载荷。请使用
`load_format="safetensors"`（推荐）或结构化 tensor dict。
`serialize_state_dict` 的 pickle 编码仅用于受信的同机 IPC，且需要在引擎进程中设置
`LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1`。HTTP 路径上的未认证 pickle 反序列化会被拒绝。

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

**注意**: flattened bucket 的 pickle 传输仅适用于受信的本地 IPC，且需要设置
`LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1`。不要将该编码 POST 到 HTTP `/update_weights`。

## 步骤 4: 唤醒引擎

权重更新后，调用API构建KV缓存，唤醒引擎，重新提供推理服务。

```python
response = requests.post(f"{BASE_URL}/wakeup", headers=headers, params=dict(tags=['kv_cache']))
assert response.status_code == 200, response.status_code
```
