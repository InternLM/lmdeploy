# Context Parallel

LMDeploy exposes two backend-specific context-parallel features:

- TurboMind context parallelism uses `--cp`.
- PyTorch decode context parallelism (DCP) uses `--dcp`.

The options are intentionally separate because their supported models and
runtime implementations differ.

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

The PyTorch backend supports decode context parallelism for FlashMLA-backed
MLA models. DCP reuses ranks from the attention tensor-parallel group and
shards each logical MLA KV sequence over a contiguous subgroup. Sparse MLA
also shards the DSA indexer cache. DCP does not launch additional model
ranks.

For example, `TP=8,DCP=4` creates the DCP groups `[0,1,2,3]` and
`[4,5,6,7]`. Inside each group, global token position `p` is stored by rank
`p % 4` at local position `p // 4`. Each rank scans only its local history.
The ranks gather query heads and merge local attention results with their
log-sum-exp statistics before reducing and scattering the output heads.
Sparse MLA additionally exchanges DSA top-k candidates.

```bash
lmdeploy serve api_server <mla-model> \
    --backend pytorch \
    --tp 4 \
    --dcp 2
```

The equivalent Python configuration is:

```python
from lmdeploy import PytorchEngineConfig, pipeline

pipe = pipeline(
    '<mla-model>',
    backend_config=PytorchEngineConfig(tp=4, dcp=2),
)
```

### Cache capacity and prefill

A physical FlashMLA cache page continues to hold 64 local tokens. With
`DCP=N`, the scheduler treats that page as one logical block spanning
`64 * N` global tokens. Physical cache bytes per rank are unchanged, while
the logical token capacity grows by approximately `N`.

DCP is decode-oriented: normal prefill attention remains tensor parallel.
Prefill still inserts MLA cache entries according to the DCP owner rule, as
well as DSA cache entries for sparse MLA. When a cached prefix is reused,
LMDeploy gathers and de-interleaves one bounded context chunk at a time. Each
chunk is attended independently, then combined with the current-token result
using its log-sum-exp statistics. The full cached prefix is therefore never
materialized on one rank. The same partition-and-merge flow is used after DSA
switches prefill from dense MLA to sparse top-k attention.

Context chunks are planned once per step from a 64 MiB KV-gather workspace
(or enough for one logical block per request, if larger), independently of
the incoming query length. Partition outputs accumulate in FP32 and are cast
back to the model dtype after the final merge. Cache sizing reserves this
workspace and the merge buffers. Under DCP, the DSA prefill logits budget also
covers local and gathered top-k candidates; completed output indices are
reserved separately.

The BF16 MLA cache is supported for dense and sparse MLA. The blocked-FP8 MLA
cache is also supported for sparse MLA. CUDA graph decode uses fixed-size
sequence-length, LSE, and collective shapes from the existing batch buckets;
sparse MLA also uses fixed-size candidate shapes.

### Current restrictions

When `dcp > 1`, the current implementation supports:

- CUDA on NVIDIA Hopper/SM90.
- FlashMLA-backed dense MLA models, or sparse MLA models with DSA top-k 512
  or 2048. Model activations must use BF16.
- Sparse MLA requires compatible DeepGEMM MQA-logits APIs, the TileLang
  sparse top-k selector, and FlashMLA support for per-row `topk_length`.
- `tp` is divisible by `dcp`, and `dcp` divides the replicated KV
  head count.
- `dp=1`, `ep=1`, and the hybrid engine role.
- MTP (`deepseek_mtp`) with dense MLA (DeepSeek V3/V3.1) or sparse MLA
  (DeepSeek V3.2/GLM DSA).
- No sliding-window attention, MemDecode,
  prefill/decode disaggregation, or external KV-cache connector.

`dcp=1` is the default and preserves the existing PyTorch execution and
cache layout without DCP collectives.
