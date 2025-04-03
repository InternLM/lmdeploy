# LMDeploy-DistServe

## Key Components
1. ​**Router Service**: Coordinates between prefill/decode engines
4. ​**Migration Manager**: Facilitates high-performance memory sharing

## Installation
```
# Inference Engine
pip install lmdeploy[all] >= 0.7.0

# Transfer Engine
pip install dlslime==0.0.1.post1
```

## Quick Start
### 1. Configure Endpoints
First deploy your prefill and decode engines.

``` shell
# Prefill Engine
CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 --role Prefill --tp 2 --cache-block-seq 32
# Decode Engine
CUDA_VISIBLE_DEVICES=2,3 lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23334 --role Decode --tp 2 --cache-block-seq 32
```

### 2. Launch Router Service

``` shell
python -m lmdeploy.disagg.router \
    --host 0.0.0.0 \
    --port 5000 \
    --prefill-endpoint http://prefill-host:port1 http://prefill-host:port2 \
    --decode-endpoint http://decode-host:port3 http://decode-host:port4
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

``` bash
ibstatus      # Verify IB device status
ibv_devinfo   # Check device capabilities
```

### Check NVSHMEM configuration:
Make sure to verify NVSHMEM installation.
