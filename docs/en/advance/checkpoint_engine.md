# Updating PyTorch weights with checkpoint-engine

LMDeploy's PyTorch engine can receive model weights from
[MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)
through CUDA IPC. The receiver is shared by checkpoint-engine's Broadcast and
P2P update modes; the difference is how the ParameterServer fills the GPU
buffer before exporting it to LMDeploy.

## Supported configuration

| Item        | Support                                                        |
| ----------- | -------------------------------------------------------------- |
| Backend     | PyTorch                                                        |
| Device      | CUDA                                                           |
| Parallelism | TP, DP, and EP; targets must contain complete TP and EP groups |
| Weights     | FP16/BF16 and PyTorch FP8/MoE loaders                          |
| Lifecycle   | Empty initialization and explicit online sleep/wakeup          |

Turbomind, non-CUDA devices, partial TP/EP-group updates, EPLB, MemDecode, and
AWQ/GPTQ validation are not supported by this integration.

Install the tested checkpoint-engine version. Broadcast only needs the base
package; the `p2p` extra installs Mooncake Transfer Engine as well:

```bash
pip install 'checkpoint-engine==0.4.2'          # Broadcast
pip install 'checkpoint-engine[p2p]==0.4.2'
```

P2P additionally requires `libnuma.so.1` (the `libnuma1` package on Ubuntu) and
a working RDMA environment. If Mooncake discovers a number of HCAs that is not
compatible with the visible GPU count, select the intended HCAs explicitly:

```bash
export PS_P2P_STORE_RDMA_DEVICES=mlx5_0,mlx5_1
```

checkpoint-engine initializes its P2P store whenever the P2P extra is
installed, including during Broadcast, so the HCA setting can also be needed
for Broadcast in that environment. Every ParameterServer process that hands a
CUDA IPC buffer to LMDeploy must run on the same physical GPU as the
corresponding LMDeploy worker. CUDA UUIDs are checked before the transfer.
This colocation requirement applies to the inference-side ParameterServer that
exports the final CUDA IPC buffer. A P2P source ParameterServer may remain on a
remote training node; an inference-side joining ParameterServer first pulls
from that source over Mooncake/RDMA and then exports locally to LMDeploy.

## Start an empty PyTorch server

Explicitly select Ray and use `--empty-init`. LMDeploy builds the model
structure but does not load weights, allocate KV cache, or run warmup:

```bash
lmdeploy serve api_server /path/to/model-config-and-tokenizer \
  --backend pytorch \
  --distributed-executor-backend ray \
  --empty-init \
  --tp 2 \
  --api-keys YOUR_API_KEY
```

For DP, configure the rendezvous shared by all DP workers before starting the
API servers:

```bash
export LMDEPLOY_DP_MASTER_ADDR=10.0.0.10
export LMDEPLOY_DP_MASTER_PORT=29500
```

EP requires the Ray executor. For example, Qwen3.5-35B-A3B with DP4/EP4 has
one local worker per API endpoint, while all four global ranks form one EP
group:

```bash
lmdeploy serve api_server /path/to/Qwen3.5-35B-A3B \
  --backend pytorch \
  --distributed-executor-backend ray \
  --empty-init \
  --tp 1 --dp 4 --ep 4 \
  --api-keys YOUR_API_KEY
```

Both Broadcast and P2P must update all four endpoints together in this
topology. Updating only one DP endpoint would omit experts owned by the other
ranks and must be rejected before transfer.

The service reports healthy-but-sleeping and rejects inference until both
weights and KV cache are ready.

Check the receiver before starting ParameterServer transfer:

```bash
curl -H 'Authorization: Bearer YOUR_API_KEY' \
  http://127.0.0.1:23333/update_weights_from_ipc
```

A ready response contains the worker CUDA UUIDs in local rank order:

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
  "worker_ranks": [0]
}
```

For multiple endpoints, the driver must derive the target group from its LMDeploy
distributed configuration. The readiness response reports runtime worker identity;
it does not describe the distributed topology.

## ParameterServer callback

Run one ParameterServer process per LMDeploy model worker. For each complete
inference-parallel group, its source rank sends the group's UUID-to-ZMQ mapping
to LMDeploy:

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

Register and gather the checkpoint using checkpoint-engine before calling
`update` as documented by its `ParameterServer` API. Split checkpoint files or
tensor names deterministically across ParameterServer ranks.

### Broadcast

Leave `ranks` unset. checkpoint-engine broadcasts the registered checkpoint to
all ParameterServer ranks, then each rank exports its buffer to its colocated
LMDeploy worker:

```python
ps.update('checkpoint-name', request_lmdeploy, ranks=None)
```

After the update succeeds, weights are marked ready but inference remains
blocked. Allocate and warm up KV cache explicitly:

```bash
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=kv_cache'
```

### P2P

Pass the complete target-rank groups to `ranks`. This selects
checkpoint-engine's Mooncake P2P path:

```python
# Update one complete TP=2 replica.
ps.update('checkpoint-name', request_lmdeploy, ranks=[0, 1])
```

For a joining ParameterServer, keep the source ParameterServer alive, load its
serialized metadata with `load_metas`, and use the joining server's local CUDA
UUID topology for the LMDeploy callback. The source pinned-memory registration
and Mooncake endpoint must remain valid until the transfer completes.

Never pass only part of an LMDeploy TP or EP group. With DP/EP, call every
relevant DP API server directly and concurrently; sleep, wakeup, and EP
inference can enter cross-rank collectives, so sequential fan-out can deadlock.
The LMDeploy proxy does not fan out management updates.

For example, DP4/EP4 P2P must pass the complete group:

```python
ps.update('checkpoint-name', request_lmdeploy, ranks=[0, 1, 2, 3])
```

## Online update

LMDeploy does not automatically drain requests for this integration. Explicitly
sleep the target, restore/rebuild its weight allocation, update it, and finally
wake KV cache:

```bash
# 1. Block new requests, drain work, and release runtime memory.
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/sleep?level=2'

# 2. Rebuild an empty CUDA weight allocation.
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=weights'

# 3. Check GET /update_weights_from_ipc, then run ps.update(...).

# 4. Rebuild KV cache and resume inference.
curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:23333/wakeup?tags=kv_cache'
```

Level 1 keeps a CPU copy of the previous weights; level 2 discards them and
rebuilds an empty model before the update.

## Failure recovery

An IPC update is not transactional. If any worker fails after transfer starts,
LMDeploy reports that weights may be partially updated and keeps inference
sleeping. Do not call `wakeup(kv_cache)` in that state. Stop and recreate every
affected DP replica, then repeat readiness and the complete update.

Check these items before investigating a hang or failure:

- every endpoint reports `ready=true`;
- ParameterServer and LMDeploy UUID groups match exactly and in rank order;
- all target ranks form complete TP and EP groups;
- source ParameterServers and metadata remain alive for P2P;
- the API key is supplied to both readiness and update requests;
- Mooncake/RDMA devices are available for P2P.

When registering safetensors under `/dev/shm`, checkpoint-engine's in-place pin
mode can remove the files. Keep it disabled unless this destructive behavior is
explicitly intended.
