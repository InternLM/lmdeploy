# Context Parallel

When the memory on a single GPU is insufficient to deploy a model, it is often deployed using tensor parallelism (TP), which generally requires `num_key_value_heads` to be divisible by `TP`. If you want to deploy with `TP > num_key_value_heads`, the kv-heads should be duplicated to meet the divisibility requirement. However, this has two disadvantages:

1. The amount of available kv_cache is halved, which reducing the maximum supported session length.
2. The maximum inference batch size is reduced, leading to lower throughput.

To address this issue, the TurboMind inference backend supports setting `attn_dp_size`, which avoids creating copies of kv-heads, but this introduces data imbalance. To eliminate data imbalance, TurboMind supports sequence parallelism, which allowing kv_cache to be stored interleaved on different cp_ranks. See the example below:

```
cp_rank=2, prompt_len=5, generation_len=4
kv_cache stored on cp_rank0: 0, 2, 4, 6, 8
kv_cache stored on cp_rank1: 1, 3, 5, 7
```

## Usage

Taking Intern-S1 / Qwen3-235B-A22B as an example, their `num_key_value_heads` is 4. If you want to deploy with `TP=8` and avoid duplication of kv_cache, you can deploy in the following way:

```
lmdeploy serve api_server internlm/Intern-S1 --tp 8 --cp 2

lmdeploy serve api_server Qwen/Qwen3-235B-A22B --tp 8 --cp 2
```
