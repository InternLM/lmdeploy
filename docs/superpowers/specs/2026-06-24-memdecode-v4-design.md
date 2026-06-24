# MemDecode v4 Design

Date: 2026-06-24
Branch: `support-memdecode-v4`

## Summary

MemDecode v4 moves away from treating the base model and memory model as one `nn.Module`. The base model should be built and executed normally. The memory model should be owned by a dedicated `MemDecodeAgent`, similar in lifecycle shape to speculative decoding, but without draft-token proposal or rejection sampling.

MemDecode and speculative decoding are mutually exclusive in v4.

The core execution order is:

1. Run the base model over the accepted-token input.
2. Run the memory model over the same accepted-token input through `MemDecodeAgent`.
3. Slice hidden states to the positions that need logits.
4. Materialize base and memory logits only for those sliced positions.
5. Fuse logits with fixed lambda or adaptive router.
6. Sample from fused logits in the base vocabulary.

## Goals

- Preserve both fusion modes from the current implementation:
  - fixed `lambda_value` fusion;
  - adaptive router fusion.
- Keep the base model as an ordinary patched model instead of wrapping it in `MemDecodeForCausalLM`.
- Give the memory model separate model, graph runner, KV cache, and state cache ownership.
- Share scheduler-owned logical block and state offsets between base and memory execution.
- Avoid full-prompt vocabulary logits for long prompts.
- Align memory logits to the base vocabulary before fusion and warn when vocab sizes differ.
- Fail early for unsupported or ambiguous configurations.

## Non-Goals

- Do not support MemDecode and speculative decoding together in v4.
- Do not add separate EP handling for the memory model. The memory model is treated as dense.
- Do not support separate TP settings for memory and base in v4. Memory uses the same `attn_tp`.
- Do not optimize long-prefill memory logits beyond the baseline sliced-logits design. Later work may compute only final-token memory logits while still filling memory KV/state for the full prompt.
- Do not support sleep/wakeup for MemDecode in v4.

## Architecture

### BaseModelAgent

`BaseModelAgent` remains the owner of normal base-model execution, sampling, scheduling integration, graph runner lifecycle, and output production.

When MemDecode is enabled, `BaseModelAgent` also owns a `MemDecodeAgent` and a fusion component. It must reject configurations that also enable speculative decoding.

### MemDecodeAgent

`MemDecodeAgent` owns the memory model path:

- build and load the memory model;
- build the memory graph runner;
- build memory KV cache and memory state cache engines;
- run memory forward on the same `ModelInputs` used by the base model;
- warm up memory execution if needed;
- reset and release memory resources during engine teardown.

It does not propose tokens, sample tokens, or run rejection sampling.

### MemDecodeFusion

Fusion should be explicit engine-side logic, not a model wrapper. It takes sliced base hidden states, sliced memory hidden states, base logits, and memory logits, then returns fused base-vocab logits plus optional routed weights.

Fixed fusion uses the configured `lambda_value`. Adaptive fusion loads the router from `router_path`, validates dimensions against the actual base and memory hidden sizes, and emits per-token log mixing weights.

## Configuration

`ModelConfig` keeps the existing MemDecode fields:

- `memory_model_path`
- `memory_model_config`
- `lambda_value`
- `adaptive_router`
- `router_path`
- `lambda_base_only_threshold`

These fields configure `MemDecodeAgent` and `MemDecodeFusion`; they should not cause the base HF architecture to become `MemDecodeForCausalLM`.

If `memory_model_path` is present, MemDecode is enabled.

The memory model uses the same `attn_tp` as the base model. EP and MoE distribution choices apply to the base model only. The memory model has its own model config and may differ in layer count, hidden size, KV-head count, head dimensions, dtype, cache shapes, and vocab size.

Memory KV-head handling must support `num_key_value_heads < tp` by using the same KV-head replication behavior as normal model config construction.

## Data Flow

For both prefill and decode, base and memory consume the same accepted token stream. The memory agent receives the same `ModelInputs` as the base forward, including:

- `input_ids`
- `seq_length`
- `history_lengths`
- `block_offsets`
- `num_ignored_history`
- chunking flags
- `state_offsets`
- multimodal and embedding metadata when present
- mrope metadata when present

The memory agent builds its own `StepContext` using `memory_model_config`, memory KV caches, and memory state caches.

The base forward should return hidden states. The memory forward should return hidden states. The engine slices hidden states to the positions that need logits before calling `get_logits()` for either model.

Sampling always uses fused logits aligned to the base vocabulary.

## Long Prompts

For long prompts, v4 must follow lmdeploy's existing chunked prefill path. The prompt is split by `max_prefill_token_num` and processed chunk by chunk.

"Memory model runs over the full prompt" means every prompt token is processed by the memory model to fill memory KV/state, but it does not mean full-prompt vocabulary logits are materialized.

Required long-prompt behavior:

- base and memory run on the same chunk boundaries;
- non-final chunks fill KV/state and do not produce user-visible sampling output;
- non-final chunks do not materialize fused vocab logits;
- final chunks compute the logits needed for sampling after hidden-state slicing;
- adaptive router uses sliced hidden/logit features, not full-prompt logits.

This avoids the current wrapper risk where full-prompt base logits and memory logits can each scale as `tokens * vocab_size`.

If an API path requests full prompt logits for a long chunked prefill while MemDecode is enabled, v4 should reject that request clearly instead of materializing full-prompt fused logits. The baseline only supports logits for emitted or sampling positions.

## Fusion

Before fusion, memory logits are aligned to the base vocab size:

- if memory vocab is larger than base vocab, truncate memory logits to base vocab;
- if memory vocab is smaller than base vocab, pad missing positions with `-inf`;
- log a warning during config/build when vocab sizes differ.

Fixed fusion uses a log-prob mixture and must handle `lambda_value` endpoints:

- `lambda_value = 0`: base-only;
- `lambda_value = 1`: memory-only after base-vocab alignment;
- intermediate values: logaddexp mixture.

Adaptive router fusion:

- requires `router_path`;
- requires a memory model;
- validates router dimensions against base and memory hidden sizes;
- uses sliced base/memory hidden states and sliced base/memory logits;
- applies `lambda_base_only_threshold` when configured;
- can return routed weights for existing routed-output plumbing.

## Cache And Eviction

The scheduler owns one logical block table and, for SSM models, one logical state id per sequence.

The base model and memory model own separate physical cache tensors:

- base KV cache engine;
- memory KV cache engine;
- base state cache engine;
- memory state cache engine.

Both models index their physical caches with the same scheduler-provided `block_offsets` and `state_offsets`.

Cache sizing must reserve memory for both models:

```text
total_block_bytes = base_block_bytes + memory_block_bytes
```

Spec decode is mutually exclusive with MemDecode in v4, so target + spec + memory cache sizing is not required.

Recompute eviction should replay both models. Eviction frees logical block and state allocation. Physical cache tensors do not need to be cleared. When the sequence is scheduled again, base forward overwrites base cache tensors and memory forward overwrites memory cache tensors at the newly allocated block/state offsets.

State cache rules:

- base and memory must either both have `states_shapes` or both have none;
- if both have SSM state, one logical `state_offset` is shared;
- base and memory state cache pools are separate physical pools indexed by that same logical state id;
- mismatched SSM presence fails during validation.

## Validation And Errors

v4 should fail early for unsupported or unsafe configurations:

- `memory_model_path` and speculative decoding cannot both be enabled.
- Base and memory SSM state presence must match.
- `lambda_value` must be in `[0, 1]`.
- Adaptive router requires `router_path`, memory model enabled, and compatible router dimensions.
- Memory cache block sizing must succeed with memory model shapes.
- Sleep and wakeup are unsupported when MemDecode is enabled and should raise a clear error.
- Full-prompt returned logits for long chunked prefill are unsupported when MemDecode is enabled and should raise a clear error.

Runtime memory forward or fusion failures should be loud. v4 should not silently fall back to base-only because that would make quality and performance measurements misleading.

Debug tensor-shape logging should be debug level or one-time initialization logging, not warning-level per-forward logging.

## Lifecycle

When MemDecode is enabled:

- engine/model release must clear memory-agent model, graph runner, KV cache, and state cache resources;
- sleep and wakeup should raise a clear unsupported error;
- base-only mode keeps existing sleep/wakeup behavior.

## Testing Scope

Tests should focus on high-risk behavior only.

Required useful tests:

- reject MemDecode plus speculative decoding;
- reject base/memory SSM mismatch;
- fusion aligns vocab correctly and handles `lambda_value` endpoints;
- long prefill does not materialize full-sequence vocab logits before slicing;
- cache sizing includes memory KV/state bytes;
- memory model with `num_key_value_heads < tp` produces valid config/cache shapes;
- sleep/wakeup raise clearly when MemDecode is enabled;
- long chunked prefill with full-prompt returned logits is rejected clearly;
- one fixed-fusion integration or smoke path runs end to end.

Avoid low-value tests for simple dataclass assignment, default `None` plumbing, or exact warning text.

## Migration From Current Branch

The existing `MemDecodeForCausalLM` wrapper should be removed or no longer used as the primary integration path.

The smoke script should move from:

```json
{"architectures": ["MemDecodeForCausalLM"]}
```

to ordinary base-model architecture plus MemDecode overrides such as:

```json
{
  "memory_model_path": "/path/to/memory",
  "lambda_value": 0.5,
  "adaptive_router": false
}
```

Adaptive mode adds `router_path` and `lambda_base_only_threshold` when needed.
