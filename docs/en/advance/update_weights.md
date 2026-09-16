# Update Weights

LMDeploy supports update model weights online for scenes such as RL training. Here are the steps to do so.

For checkpoint-engine Broadcast and Mooncake P2P updates, see
[Updating PyTorch weights with checkpoint-engine](./checkpoint_engine.md).

`POST /update_weights` does not accept pickle payloads. Send
`load_format="safetensors"` (recommended) or a structured dict of tensors.
Pickle encoding from `serialize_state_dict` is only for trusted same-node IPC
when `LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1` is set on the engine process.
Unauthenticated pickle deserialization over HTTP is rejected.

## Step 1: Launch server

For pytorch backend you have to add `--distributed-executor-backend ray`.

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --distributed-executor-backend ray # for pytorch backend
```

## Step 2: Offloads weights & kv cache

Before update model weights, the server should offloads weights and kv cache.

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

## Step 3: Update weights

Split model weights into multi segments and update through `update_weights` endpoint.

```python
segmented_state_dict: List[Dict[str, torch.Tensor]] = ...
num_segment = len(segmented_state_dict)
for seg_idx in range(num_segment):
    serialized_data = serialize_named_tensors_safetensors(segmented_state_dict[seg_idx])
    data = dict(serialized_named_tensors=serialized_data, load_format='safetensors', finished=seg_idx == num_segment-1)
    response = requests.post(f"{BASE_URL}/update_weights", headers=headers, json=data)
    assert response.status_code == 200, f"response.status_code = {response.status_code}"

```

PyTorch also supports receiving weights through
`POST /update_weights_from_distributed` (NCCL) and
`POST /update_weights_from_ipc` (checkpoint-engine). Those paths do not pickle
HTTP bodies.

**Note**: Flattened-bucket pickle transfer remains available only for trusted
local IPC after setting `LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS=1` on the engine.
Do not post that encoding to HTTP `/update_weights`.

## Step 4: Wakeup server

After update model weights, the server should onloads kv cache and provide serving again with the new updated weights.

```python
response = requests.post(f"{BASE_URL}/wakeup", headers=headers, params=dict(tags=['kv_cache']))
assert response.status_code == 200, response.status_code
```
