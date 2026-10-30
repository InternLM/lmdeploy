# Use LMCache with TurboMind

TurboMind can use [LMCache](https://github.com/LMCache/LMCache) to store KV cache
outside GPU memory and retrieve it for later requests. The integration performs
LOOKUP, STORE and RETRIEVE through an LMCache multiprocess server.

Local prefix caching and LMCache can be enabled together. Local prefix caching
reuses GPU-resident KV; LMCache can supply cached tokens that are unavailable
locally. Reusing external KV can reduce repeated prefill computation, while
lookup and data transfers also have a cost. Measure performance with your own
workload, especially when local GPU cache is already sufficient.

## Requirements and installation

Use Linux, NVIDIA GPUs and an LMDeploy build with LMCache support; see the
[installation guide](../get_started/installation.md). LMCache and TurboMind must
run on the same host and use the same physical GPUs.

In a Python environment with PyTorch and a compatible CUDA toolkit installed,
install the TurboMind adaptation based on LMCache 0.5.5:

```bash
git clone --branch tm https://github.com/irexyc/lmcache.git
cd lmcache
git checkout a41e1d0a5b8624a0bc7aa2fff468984845a4ca10
python -m pip install -r requirements/build.txt
LMCACHE_CUDA_MAJOR=12 python -m pip install -e . --no-build-isolation
```

The commands above use CUDA 12. For CUDA 13, set `LMCACHE_CUDA_MAJOR=13`.

## Start the services

### 1. Start LMCache

In the first terminal:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
lmcache server \
  --host 127.0.0.1 \
  --port 5558 \
  --chunk-size 256 \
  --l1-size-gb 512 \
  --eviction-policy LRU
```

Wait for the server to report that it is running. `--l1-size-gb 512` configures a
512 GiB host-memory cache; reduce this value if the host has less available
memory. The selected GPUs must be available for this deployment.

### 2. Start the TurboMind API server

In a second terminal, use the same GPU selection:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=Qwen/Qwen3-30B-A3B
lmdeploy serve api_server "$MODEL_PATH" \
  --backend turbomind \
  --tp 4 \
  --enable-prefix-caching \
  --session-len 65536 \
  --lmcache-addr tcp://127.0.0.1:5558
```

`MODEL_PATH` can also be a local model directory. The API server listens on
port 23333 by default. The 65,536-token session limit accommodates a 32K initial
prompt and additional conversation turns; choose a limit suitable for your
model and workload. No `--async` argument is needed to use the default value 1.

Send requests through the ordinary
[OpenAI-compatible API](../llm/api_server.md#restful-api). Clients do not need
additional LMCache parameters. To disable LMCache, omit `--lmcache-addr` when
starting the API server; local prefix caching is controlled independently.

## Use the Python pipeline

Keep the LMCache server running and use the pipeline as an alternative to the
API server. Set `CUDA_VISIBLE_DEVICES=0,1,2,3` before launching Python.

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

backend_config = TurbomindEngineConfig(
    tp=4,
    enable_prefix_caching=True,
    session_len=65536,
    lmcache_addr='tcp://127.0.0.1:5558',
)

with pipeline('Qwen/Qwen3-30B-A3B', backend_config=backend_config) as pipe:
    response = pipe(
        'Explain how KV caching accelerates language model inference.',
        gen_config=GenerationConfig(max_new_tokens=64),
    )
    print(response.text)
```

Omitting `lmcache_addr`, or setting it to `None`, disables the external cache.
See the [pipeline guide](../llm/pipeline.md) for other inference options.

## Configuration and limitations

| Setting                                             | Purpose                                                                                             |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `--lmcache-addr` / `lmcache_addr`                   | LMCache ZeroMQ endpoint. Unset by default, which disables LMCache.                                  |
| `--enable-prefix-caching` / `enable_prefix_caching` | Local GPU prefix caching. Independent of LMCache.                                                   |
| LMCache `--chunk-size`                              | External cache transfer granularity in tokens. Use 256 with the default TurboMind block size of 64. |
| LMCache `--l1-size-gb`                              | Host-memory cache capacity in GiB; separate from TurboMind's GPU cache capacity.                    |
| LMCache `--eviction-policy`                         | L1 eviction policy; the examples use LRU.                                                           |

The LMCache chunk size must be divisible by TurboMind's logical block size
(`cache_block_seq_len * cp`, with `cp=1` in this guide). For GDN models, TurboMind
also captures the recurrent state at chunk boundaries; no additional GDN flag
is required. See [TurboMind configuration](../inference/turbomind_config.md) for
GPU cache sizing and prefix-cache options.

Requests for perplexity or all-token logits/hidden states bypass external
caching because they need prompt computation. Native prefix caching has its own
restrictions for these output modes. Dynamic NTK requests with a request-dependent
RoPE base also bypass external caching. This guide covers text inputs; multimodal
fingerprint collision handling remains incomplete.

Start a fresh LMCache instance when changing the model, weights, TP configuration
or cache layout. This integration does not provide compatibility checks for
historical entries from a different deployment.
