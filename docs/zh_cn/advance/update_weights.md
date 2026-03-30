# 权重更新

LMDeploy支持在线权重更新，方便RL训练等场景下的使用。以下是权重更新的步骤：

## 步骤 1: 启动服务

For pytorch backend you have to add `--distributed-executor-backend ray`.

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --distributed-executor-backend ray # for pytorch backend
```

## 步骤 2: 卸载权重和KV缓存

在权重更新前，需要调用API卸载权重和KV缓存，使推理引擎处于可更新状态：

```python
from lmdeploy.utils import serialize_state_dict
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
    serialized_data = serialize_state_dict(segmented_state_dict[seg_idx])
    data = dict(serialized_named_tensors=serialized_data, finished=seg_idx == num_segment-1)
    response = requests.post(f"{BASE_URL}/update_weights", headers=headers, json=data)
    assert response.status_code == 200, f"response.status_code = {response.status_code}"

```

**注意**: 对于pytorch推理后端，lmdeploy还支持扁平化桶张量(flattened bucket tensor)传输方式:

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
