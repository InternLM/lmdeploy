# Context Parallel

LMDeploy exposes two backend-specific context-parallel features:

- TurboMind context parallelism uses `--cp`.
- PyTorch decode context parallelism (DCP) uses `--dcp`.

## TurboMind context parallelism

When the memory on a single GPU is insufficient to deploy a model, it is often deployed using tensor parallelism (TP), which generally requires `num_key_value_heads` to be divisible by `TP`. If you want to deploy with `TP > num_key_value_heads`, the kv-heads should be duplicated to meet the divisibility requirement. However, this has two disadvantages:

1. The amount of available kv_cache is halved, which reducing the maximum supported session length.
2. The maximum inference batch size is reduced, leading to lower throughput.

To address this issue, the TurboMind inference backend supports setting `attn_dp_size`, which avoids creating copies of kv-heads, but this introduces data imbalance. To eliminate data imbalance, TurboMind supports sequence parallelism, which allowing kv_cache to be stored interleaved on different cp_ranks. See the example below:

```
cp_rank=2, prompt_len=5, generation_len=4
kv_cache stored on cp_rank0: 0, 2, 4, 6, 8
kv_cache stored on cp_rank1: 1, 3, 5, 7
```

Under context parallelism, `cache_block_seq_len` remains the physical number of tokens stored by one rank in a k/v cache block. The scheduler treats the corresponding logical block as `cache_block_seq_len * cp` global tokens. Therefore k/v block memory on each rank is unchanged, while full-block prefix reuse and read-only cache boundaries use the larger global span.

### Usage

Taking Intern-S1 / Qwen3-235B-A22B as an example, their `num_key_value_heads` is 4. If you want to deploy with `TP=8` and avoid duplication of kv_cache, you can deploy in the following way:

```
lmdeploy serve api_server internlm/Intern-S1 --tp 8 --cp 2

lmdeploy serve api_server Qwen/Qwen3-235B-A22B --tp 8 --cp 2
```

## PyTorch decode context parallelism

PyTorch DCP distributes replicated KV cache across existing TP ranks, increasing
effective cache capacity without additional GPUs. It supports FlashMLA-backed
dense MLA, sparse DSA, and GQA with replicated KV heads.

```bash
lmdeploy serve api_server <mla-model> --backend pytorch --tp 4 --dcp 2

# Both Qwen3 models have four KV heads; TP=8 replicates each head twice.
lmdeploy serve api_server Qwen/Qwen3-30B-A3B --backend pytorch --tp 8 --dcp 2
lmdeploy serve api_server Qwen/Qwen3-235B-A22B --backend pytorch --tp 8 --dcp 2
```

For Python, use `PytorchEngineConfig(tp=4, dcp=2)`. The default `dcp=1`
disables DCP.

### Requirements and supported features

- MLA requires NVIDIA Hopper/SM90 GPUs, FlashMLA, and BF16 activations.
  Sparse DSA also requires compatible DeepGEMM and TileLang top-k kernels
  (top-k 512 or 2048).
- GQA uses Triton for decode and FlashAttention-3, when available, for prefill,
  with a Triton prefill fallback. It supports BF16/FP16 activations and
  unquantized or per-tensor FP8 KV cache (`--quant-policy fp8` for E4M3,
  `--quant-policy fp8_e5m2` for E5M2). Cached-prefix chunks are dequantized
  before gathering. INT8, INT4, TurboQuant, ALiBi, and attention sinks are not supported.
- `dcp` must divide the attention TP size and replicated KV-head count;
  `dp=1` and `ep=1` are required.
- Prefix caching and `deepseek_mtp` are supported for compatible MLA models,
  including DeepSeek V3/V3.1, DeepSeek V3.2, and GLM DSA.
- GQA prefill retains local query heads and gathers cached K/V in bounded
  chunks. Decode gathers query heads within each replicated-KV group and
  merges shard outputs using their softmax normalization factors.
- BF16 KV cache is supported. Sparse MLA also supports FP8 KV cache via
  `--quant-policy fp8`.
- Sliding-window attention, MemDecode, prefill/decode disaggregation,
  external KV-cache connectors, and the TileLang attention backend are not supported.
- GQA DCP rejects active sliding windows at the individual layer boundary,
  including mixed local/global models such as Gemma 3. Non-causal GQA attention
  and SDAR/dLLM block-sparse attention (`dllm_block_length > 1`) are also rejected.
- DFlash with `dcp > 1` is unsupported: its draft cache writes and non-causal
  block attention do not implement DCP. Llama EAGLE3 is rejected when its
  unsharded draft cache would reuse the DCP target's incompatible page table.
  These combinations must use `dcp=1`. Compatible speculative methods that
  share the target's cache layout, such as `deepseek_mtp`, remain supported.

### Communication policy and compatibility

`--communication-backend nccl` is the default and uses native collectives unless
an existing operation-specific environment variable explicitly opts into an
optimization. `--communication-backend auto` opts into eligible symmetric-memory
gathers and TP all-reduce, with native fallback for unsupported inputs/topologies.

The existing `LMDEPLOY_ENABLE_FLASHINFER_ALLREDUCE`,
`LMDEPLOY_ENABLE_SYMM_MEM_ALLREDUCE`, and `LMDEPLOY_ENABLE_SYMM_MEM_LMHEAD`
options retain their TP/LM-head scope. An explicit FlashInfer opt-in takes
precedence over `auto` for TP all-reduce and preserves DeepSeek V3.2's
all-reduce + residual + RMSNorm dispatch. Enabling both legacy all-reduce
providers is an error. Legacy TP options do not enable DCP gather optimization.

`LMDEPLOY_SYMM_MEM_LMHEAD_MAX_MB` continues to limit the LM-head gather workspace
per rank (default 64 MiB, minimum 1 MiB), including with `auto`. It does not set
the size of DCP query/prefix workspaces. Communication settings must agree
across ranks in each group.

### GQA FP8 prefill precision

DCP prefill uses the current chunk's unquantized K/V and combines it with cached
prefix K/V, dequantized using the cache's per-tensor scales. Decode reads all K/V
from the cache. This follows the MLA DCP prefill strategy, but differs from
ordinary GQA prefill, which reads both current and prefix K/V from the cache.
Consequently, enabling DCP, changing prefill chunk boundaries, or reusing a
prefix can change which tokens undergo FP8 rounding; bitwise-equivalent outputs
are not promised, and the difference is not evidence of a communication error.

`tests/pytorch/test_dcp_gqa_reference.py` covers full/chunked prefill, reuse of
cached prefix pages, and single/multi-token decode with BF16, E4M3, and E5M2.
It checks cache ownership and compares against both an independent FP32
attention oracle with the same K/V precision and the non-DCP GQA path. For its
fixed bounded inputs, the oracle tolerance is `rtol=atol=0.02`; the non-DCP and
chunk-boundary comparison tolerances are 0.02 (BF16), 0.04 (E4M3), and 0.08
(E5M2). These are regression thresholds for the fixture, not model-accuracy
guarantees.
