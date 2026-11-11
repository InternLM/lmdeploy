# Update Weights

LMDeploy supports update model weights online for scenes such as RL training. Here are the steps to do so.

For checkpoint-engine Broadcast and Mooncake P2P updates, see
[Updating PyTorch weights with checkpoint-engine](./checkpoint_engine.md).

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

## Failure isolation and readiness

While an update is in flight the model must never silently become
inference-ready with partial weights. LMDeploy enforces:

- **One transaction at a time.** The engine keeps a single weight-state
  machine (`ready` / `updating` / `poisoned`); concurrent update requests
  serialize behind it and a bucket of a different transaction is refused
  with HTTP 409. The executor-side guard is owned by the thread performing
  the actual write; cancelling the awaiting caller does not release it.
- **Failures latch (`poisoned`, terminal).** A failed, partially written or
  cancelled update makes the engine unavailable (`/wakeup`, further updates
  and inference routes are refused) until the engine is rebuilt from a
  complete checkpoint. A later successful bucket cannot prove which
  parameters a partial write left behind.
- **Finalization gates readiness.** Without the final bucket
  (`finished=true`) the engine stays sleeping; neither sleep nor a
  weights-only wakeup can bypass the missing finalization.
- **Readiness is re-checked at inference time.** A request admitted before
  an update but still queued when the engine becomes unavailable is refused
  through the existing request-handle path and its session is released.
- **Strict worker results.** A collective whose per-worker results are
  missing or malformed is treated as a failure, not reduced into a fake
  success.

The successful `/update_weights` response remains JSON `null`;
`/update_weights_from_ipc` and `/update_weights_from_distributed` keep their
`{success, message}` schema. A distributed `update_id` is optional: explicit
IDs enable replay and cross-transaction checks for callers that want them.

## Deployment settings

`LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS` (default `180`, positive and finite,
checked before each transfer) bounds how long an HTTP update request waits for
its bucket transfer. Exceeding it does not fail the engine: the shielded
background write keeps running and the weight state resolves by itself, so the
response reports the transfer as still in flight and `/health` shows the
resolved state (`ready` or `poisoned`) once the write finishes.

Distributed updates may carry an optional `update_id`: consecutive buckets
sharing one id form a single transaction, a committed id can never be reused
(protecting against lost-response retries re-committing a lone final bucket),
and implicit no-id callers keep their legacy multi-bucket semantics. IPC
updates still require their existing CUDA/checkpoint-engine dependencies.
