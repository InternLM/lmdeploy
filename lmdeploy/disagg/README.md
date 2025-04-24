# LMDeploy-DistServe

## Key Components

1. ​**Router Service**: Coordinates between prefill/decode engines
2. ​**Migration Manager**: Facilitates high-performance memory sharing

## Installation

```
# Inference Engine
pip install lmdeploy[all] >= 0.7.0

# Transfer Engine
pip install dlslime==0.0.1.post2
```

## Quick Start

### 1. Configure Endpoints

First deploy your prefill and decode engines.

```shell
# Prefill Engine
CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --role Prefill --tp 2 --cache-block-seq 32
# Decode Engine
CUDA_VISIBLE_DEVICES=2,3 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23334 --role Decode --tp 2 --cache-block-seq 32
```

### 2. Launch Router Service

```shell
lmdeploy serve proxy
    --server-name 10.130.8.139
    --server-port 5000
    --routing-strategy "min_expected_latency"
    --serving-strategy DistServe
    --log-level INFO
```

## API Usage

```shell
# API Invoke
curl -X POST "http://localhost:5000/v1/completions" \
-H "Content-Type: application/json" \
-d '{"model": "internlm/internlm2_5-7b-chat", "temperature":0, "prompt": "Shanghai is a city that ", "max_tokens": 16, "stream": false}'
# Output
{"id":"2","object":"text_completion","created":1743662400,"model":"/nvme1/majinming/hub/models--internlm--internlm2_5-7b-chat/snapshots/4434a5ffc2582f9d5ac45085043ed3e3264f0a9b","choices":[{"index":0,"text":" is very famous for its skyscrapers. It is also a city","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":7,"total_tokens":23,"completion_tokens":16}}
```

## Trouble Shooting

### RDMA Connection Failed:

```bash
ibstatus      # Verify IB device status
ibv_devinfo   # Check device capabilities
```

### Check NVSHMEM configuration:

Make sure to verify NVSHMEM installation.

## Fault tolerance

### CacheFree Issue​​

When the ​​Decode Engine​​ completes migration, it sends a ​​FreeCache​​ request to the ​​Prefill Engine​​. However, if the connection fails or the Decode Engine encounters an exception, ​​Cache Free may fail​​, leading to ​​memory leaks​​. Future improvements may include:

- ​​Exception monitoring in the Proxy​​ to automatically release unreferenced memory.
- ​​Adding a timeout mechanism​​ to force cache release if a response is delayed.
  ​​

### ConnectionPool Issue​​

Currently, if the ​​Proxy disconnects​​, the connection pool must be ​​warmed up again​​. A future enhancement could involve:

A ​​dedicated connection pool management server​​ (e.g., using ​​Raft-based tools like ETCD​​, as mentioned in ​​Mooncake​​) to improve ​​connection discovery​​ and avoid repeated warmups.
