# Prefix caching (PyTorch)

This guide covers local prefix caching in the PyTorch backend and the hybrid
state-checkpoint policies shared with Mooncake Store. Local caching is enabled
with `PytorchEngineConfig.enable_prefix_caching=True`; Mooncake Store must be
configured separately through `kv_transfer_config`.
For TurboMind, see its [prefix caching settings](../inference/turbomind_config.md#prefix-caching-switch).

## Prefill checkpoints for hybrid attention models

Hybrid attention models need both full-attention KV blocks and the
linear-attention state at a reusable prefix boundary.

- `prefill_save_alignment` (`--prefill-save-alignment`, default 1024) controls
  prefill state-checkpoint boundaries for both local prefix caching and Mooncake
  Store. It must be a positive multiple of the cache block size.
  With alignment 512 and a token budget of 8192,
  an uncached 800-token text prompt uses two chunks: 512 and 288 tokens,
  including when both caches are enabled. Full-attention-only models ignore this
  option. Alignment does not require a checkpoint at every multiple.
- `prefix_cache_state_budget` reserves extra checkpoint slots and uses additional
  memory. Zero adds no reserved slots; idle runtime slots may still be reused.

### Save only the last checkpoint

`prefill_save_last_only=True` (`--prefill-save-last-only`, disabled by default)
saves only the last reusable boundary aligned to `prefill_save_alignment` in
each input turn. The boundary leaves at least one input token and moves back
if needed to avoid splitting multimodal inputs, including the shifted MTP
input. Earlier chunks use their normal token budget; the chunk reaching this
boundary saves the local state checkpoint and sends all unsaved full-attention
KV blocks through that boundary, together with its state snapshot, to Mooncake
Store. Each cache must be enabled separately.

For example, with alignment explicitly set to 8192 and 20,000 input tokens,
the save is at 16,384; if an image spans \[16,000, 18,000), it moves back to 8192.
A 32,768-token input saves at 24,576, leaving a suffix for prefix reuse.
No save is made if no positive safe boundary exists or a cache hit already
reaches it. Earlier turns' checkpoints remain subject to normal eviction,
and unavailable snapshot slots retain the existing skip behavior.

This policy reduces checkpoint copies and Store submissions for append-only
agent conversations, at the cost of fewer intermediate prefixes for edited
or branching conversations. Full-attention-only models ignore it.

Supported MTP methods use token lookahead instead of block rewind for both
local and Store matches, so they do not require extra KV blocks beyond the
saved state boundary for remote reuse.

## GLM-5.3-Flash

GLM-5.3-Flash can reuse prefix caches with ordinary autoregressive decoding or
[`deepseek_mtp`](./spec_decoding.md#deepseek-mtp).
Enable `PytorchEngineConfig.enable_prefix_caching` in either mode.
The following text-only example uses TP4; choose a TP size and cache budget that
fit your hardware. Set `speculative_config=None` to disable MTP without disabling
prefix caching.

```python
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

if __name__ == '__main__':
    engine = PytorchEngineConfig(
        tp=4,
        max_batch_size=4,
        enable_prefix_caching=True,
        prefix_cache_state_budget=8,
        max_prefill_token_num=512,
        language_model_only=True,
    )
    spec = SpeculativeConfig(method='deepseek_mtp', num_speculative_tokens=3)
    with pipeline('/path/to/GLM-5.3-Flash', backend_config=engine,
                  speculative_config=spec) as pipe:
        notes = 'A rectangle has four sides and four right angles. ' * 80
        prompt = notes + '\nHow many sides does a rectangle have?'
        generation = GenerationConfig(max_new_tokens=128, do_sample=False)
        cold = pipe(prompt, gen_config=generation)
        warm = pipe(prompt, gen_config=generation)
        print(cold.cached_tokens, warm.cached_tokens)
```

### Configuration limits

- Cache blocks contain 64 tokens. If setting `num_gpu_blocks` manually, count
  these blocks rather than KPool entries.
- Positive values of `prefix_cache_decode_state_interval` must be multiples of 64.
  With `deepseek_mtp`, decode checkpoints require one draft layer without state caches.
- PD migration is not supported.

The native FP8 CUDA path also supports DP/EP serving, for example `--tp 2 --dp 2 --ep 4`. Enable text-prefill PCG with `--piecewise-cudagraph-max-tokens 512`
and `--max-prefill-token-num 512`. Startup warmup prepares these plans; skipping
warmup leaves unprepared prefills eager. Vision-bearing chunks also fall back to
eager execution. MTP draft prefill remains eager, while supported decode steps
use the ordinary CUDA graph path. These choices are independent of prefix caching.

Inspect `Response.cached_tokens` to verify actual reuse. For aggregate local
and Store hit rates, see [cache hit metrics](./metrics.md).
Changing prefill, PCG padding, batch shapes or parallel topology can change
FP8 generation trajectories; compare task accuracy and MTP acceptance as well
as cache hits rather than assuming bitwise-identical outputs or a guaranteed speedup.

## Sharing Mooncake Store across instances

Start one [Mooncake master](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html).
On each node, point `MOONCAKE_CONFIG_PATH` to a JSON file readable by every worker. For example, with RDMA:

```json
{
  "master_server_address": "store-master:50051",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "global_segment_size": "4GB",
  "local_buffer_size": "4GB"
}
```

Use the same reachable master address (not `127.0.0.1`). With Mooncake RDMA auto-discovery enabled, omit `device_name`; for manual selection, set it to the local RDMA device name(s).
Both sizes are **per Store worker**. With eight workers on one node, these defaults contribute 32 GiB to the shared cache and budget another 32 GiB for local transfer buffers.
For replicas with the same weights and cache layout, pass the same connector settings to each PyTorch server:

```shell
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"model_name":"my-model","cache_prefix":"run-42"}}'
```

### RL weight updates

For colocated RL, add these options to `kv_connector_extra_config`:

- Set `rl_weight_versioning=True` and give the initial weights a nonempty
  `weight_version`. Never reuse a version for different weights.
- For cross-replica reuse, set `rl_share_across_engines=True` and use the same
  version on all replicas. After sleep, commit the update before waking the KV cache.
- To cache routed-expert IDs alongside KV, set `store_routed_experts=True` on
  a model that supports returning them. This uses extra GPU memory on the output rank.

Keep `cache_prefix` fixed across [Ray weight updates](./update_weights.md).
For each new set of weights, pass its `weight_version` (e.g., `step-1`) only in the
final `/update_weights` request (`finished=True`). Use the same version on all
ranks and replicas that share the cache. If all ranks in an engine omit it,
the engine generates a fresh private version instead of reusing the previous one.
A rejected version does not roll back weights already loaded.
