# Update Weights

LMDeploy supports update model weights online for scenes such as RL training. Here are the steps to do so.

## Step 1: Launch server

For pytorch backend you have to add `--distributed-executor-backend ray`.

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --distributed-executor-backend ray # for pytorch backend
```

## Step 2: Offloads weights & kv cache

Before update model weights, the server should offloads weights and kv cache.

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

## Step 3: Update weights

Split model weights into multi segments and update through `update_weights` endpoint.

```python
segmented_state_dict: List[Dict[str, torch.Tensor]] = ...
num_segment = len(segmented_state_dict)
for seg_idx in range(num_segment):
    serialized_data = serialize_state_dict(segmented_state_dict[seg_idx])
    data = dict(serialized_named_tensors=serialized_data, finished=seg_idx == num_segment-1)
    response = requests.post(f"{BASE_URL}/update_weights", headers=headers, json=data)
    assert response.status_code == 200, f"response.status_code = {response.status_code}"

```

**Note**: For pytorch backend, lmdeploy also supports flattened bucket tensors:

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

## Step 4: Wakeup server

After update model weights, the server should onloads kv cache and provide serving again with the new updated weights.

```python
response = requests.post(f"{BASE_URL}/wakeup", headers=headers, params=dict(tags=['kv_cache']))
assert response.status_code == 200, response.status_code
```
