# 序列并行

在单卡显存不足以部署模型的时候，通常会以 `TP` 的方式进行部署，而这一般要求 `num_key_value_heads` 被 `TP` 整除。如果要以 `TP > num_key_value_heads` 的方式进行部署，需要创建 kv-heads 的副本，以满足整除需求。但是这样会有两个缺点：

1. 可用的 kvcache 数量减半，进而减少请求最大推理长度
2. 降低推理的最大 batch 数量，减少吞吐量。

为了解决这个问题，TurboMind 推理后端支持设置 `attn_dp_size`，避免了创建 kv-heads 的副本，但是这会引入数据的不均衡性。为了消除数据的不均衡，TurboMind 支持了序列并行，支持将 kv_cache 交错存储到不同的 cp_rank 上，例如

```
cp_rank=2, prompt_len=5, generation_len=4
kv_cache stored on cp_rank0: 0, 2, 4, 6, 8
kv_cache stored on cp_rank1: 1, 3, 5, 7
```

## 使用说明

以 `Intern-S1` / `Qwen3-235B-A22B` 为例，他们的 `num_key_value_heads` 为 4，若要用 `TP=8` 的方式部署，并避免 kv_cache 的拷贝，可以用如下的方式部署

```
lmdeploy serve api_server internlm/Intern-S1 --tp 8 --cp 2

lmdeploy serve api_server Qwen/Qwen3-235B-A22B --tp 8 --cp 2
```
