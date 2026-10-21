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

MTP configurations that rewind cache hits by full blocks still require KV
beyond the saved state boundary for remote reuse. This policy does not save
that suffix, so the latest state may be unusable remotely without those blocks
already in Store. GLM MTP with local prefix caching uses token lookahead
instead of block rewind and does not require that extra Store margin.

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
- With MTP, keep `prefix_cache_decode_state_interval=0`. Without MTP, positive
  values must be multiples of 64.
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
