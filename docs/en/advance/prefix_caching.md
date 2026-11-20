# Prefix caching (PyTorch)

When requests start with the same tokens, LMDeploy can reuse the cached prefix
and prefill only the remaining input. Enable local reuse with
`PytorchEngineConfig(enable_prefix_caching=True)` or `--enable-prefix-caching`.
Mooncake Store is optional and configured separately through `kv_transfer_config`.

Check `Response.cached_tokens` on a repeated prompt to see how many tokens were
reused. For aggregate local and Store hit rates, see [cache metrics](./metrics.md).
For TurboMind, see its [prefix caching settings](../inference/turbomind_config.md#prefix-caching-switch).

## Checkpoints for hybrid attention models

A hybrid model can reuse a prefix only if it has both the full-attention KV
blocks and the linear-attention state at the same boundary. The following
settings control where LMDeploy saves that state:

| Setting                                    | When to use it                                                                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prefill_save_alignment` (default `1024`)  | Sets eligible prefill boundaries for local caching and Store. Use a positive multiple of the cache block size; not every boundary is saved.                               |
| `prefill_save_last_only` (default `False`) | For append-only conversations, save only the last safe aligned boundary per input turn. This reduces saves but leaves fewer reusable prefixes if a conversation branches. |
| `prefix_cache_state_budget` (default `0`)  | Reserve extra GPU state slots for local checkpoints. With no extra slots, checkpoints can still use idle runtime slots.                                                   |

For example, with alignment 8192 and last-only saving, a 20,000-token prompt
saves at 16,384. If that boundary splits an image, it moves back to 8192.
The boundary must leave at least one input token. These prefill settings do not
apply to full-attention-only models.

To checkpoint generated tokens too, set `prefix_cache_decode_save_first=True`
for the first safe aligned boundary at or after the prompt. Set
`prefix_cache_decode_state_interval` to a positive multiple of the cache block
size for periodic checkpoints; the default of zero disables periodic saves.

**GLM-5.3-Flash** uses 64-token cache blocks and supports prefix reuse with or
without [DeepSeek MTP](./spec_decoding.md#deepseek-mtp). Hybrid Store and decode
checkpoints with MTP require a single stateless draft layer. Prefill/decode
disaggregation is not supported for this model.

## RL weight updates with Mooncake Store

Cached prefixes from one set of weights must not be reused with another. For
colocated RL updates, configure `KVTransferConfig` with
`kv_connector='MooncakeStoreConnector'` and `kv_role='kv_both'`. Alongside your
Store connection settings, set these options in `kv_connector_extra_config`:

- Set `rl_weight_versioning=True` and provide a nonempty initial `weight_version`
  that identifies the loaded weights. Never reuse a version for different weights.
- To share across replicas, set `rl_share_across_engines=True` (default `False`).
  The trainer must coordinate versions across replicas. After sleep, each engine
  must commit an update before waking its KV cache.
- If the workload needs routed-expert IDs, set `store_routed_experts=True` on a
  model that supports collecting them. This reserves extra GPU memory on the
  output rank; missing or incomplete route blocks are recomputed.

Follow the [Ray weight-update procedure](./update_weights.md): drain requests,
sleep, wake the weights, update them, then wake the KV cache. On the final
`/update_weights` request (`finished=True`), every rank must supply the same
nonempty `weight_version`, or all must omit it (or use `None`) for a fresh
engine-private cache. A rejected version does not roll back weights already loaded.
