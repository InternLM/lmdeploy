# LMDeploy-DistServe

## Key Components

1. ​**Router Service**: Coordinates between prefill/decode engines
2. ​**Migration Manager**: Facilitates high-performance memory sharing

## Installation

```
# Inference Engine
pip install lmdeploy[all] >= 0.7.0

# Transfer Engine
pip install dlslime>=0.0.2
```

## Quick Start

A PD disaggregated deployment of internlm2_5-7b-chat is shown below:

### 1. Launch Router Service

```shell
lmdeploy serve proxy --server-name 0.0.0.0 --server-port 8000 --routing-strategy "min_expected_latency" --serving-strategy DistServe --log-level INFO
```

LMDeploy-DistServe support both NVLink and RDMA for kvcache transferring from Prefill Engine to Decode Engine. RDMA is default model. Set `--migration-protocol NVLink` for NVLink transport.

### 2. Configure Endpoints

First deploy your prefill and decode engines.

```shell
# Prefill Engine
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --role Prefill --proxy-url http://0.0.0.0:8000 --backend pytorch
# Decode Engine
CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23334 --role Decode --proxy-url http://0.0.0.0:8000 --backend pytorch
```

By now, only **Pytorch backend** supports PD Disaggregation.

## API Usage

```shell
# API Invoke
curl -X POST "http://localhost:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{"model": "internlm/internlm2_5-7b-chat", "temperature":0, "prompt": "Shanghai is a city that ", "max_tokens": 16, "stream": false}'
# Output
{
  "id":"2",
  "object":"text_completion",
  "created":1743662400,"
  model":"internlm/internlm2_5-7b-chat",
  "choices":[
    {
      "index":0,
      "text":" is very famous for its skyscrapers. It is also a city","logprobs":null,"finish_reason":"length"
    }
  ],
  "usage": {
    "prompt_tokens":7,"total_tokens":23,"completion_tokens":16
  }
}
```

## Trouble Shooting

### RDMA Connection Failed:

Make sure ibverbs is correctly installed:

```
# on Ubuntu
sudo apt install libibverbs-dev
# on CentOS
sudo yum install ibverbs-devel
```

```bash
ibstat        # Verify IB device status
ibv_devinfo   # Check device capabilities
```

### Check GPU Direct RDMA:

By now, lmdeploy-distserve use GPUDirect RDMA to perform KVTransfer. Make sure GPUDirect RDMA Driver is loaded to kernel.

```bash
lsmod | grep nv_peer_mem
# GPUDirect RDMA info will be printed If GPUDirect RDMA is correctly loaded.
```

### Connection Pool

Currently, if the ​​Proxy disconnects​​, the connection pool must be ​​warmed up again​​. A future enhancement could involve:

A ​​dedicated connection pool management server​​ (e.g., using ​​Raft-based tools like ETCD​​, as mentioned in ​​Mooncake​​) to improve ​​connection discovery​​ and avoid repeated warmups.

### Proxy

Do not add an engine nodes to **different proxy** because it is not supported and is not considered as a right usage by now.
