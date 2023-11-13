# turbomind config

TurboMind 是 LMDeploy 的推理引擎，在用它推理 LLM 模型时，需要把输入模型转成 TurboMind 模型。在 TurboMind 的模型文件夹中，除模型权重外，TurboMind 模型还包括其他一些文件，其中最重要的是和推理性能息息相关的配置文件`triton_models/weights/config.ini`。

以 `llama-2-7b-chat` 模型为例，它的`config.ini`内容如下：

```toml
[llama]
model_name = llama2
tensor_para_size = 1
head_num = 32
kv_head_num = 32
vocab_size = 32000
num_layer = 32
inter_size = 11008
norm_eps = 1e-06
attn_bias = 0
start_id = 1
end_id = 2
session_len = 4104
weight_type = fp16
rotary_embedding = 128
rope_theta = 10000.0
size_per_head = 128
group_size = 0
max_context_token_num = 4
step_length = 1
max_batch_size = 64
cache_max_entry_count = 0.5
cache_block_seq_len = 128
cache_chunk_size = 1
use_context_fmha = 1
quant_policy = 0
max_position_embeddings = 2048
use_dynamic_ntk = 0
use_logn_attn = 0
```

在这份配置中，可调参数为：

```toml
max_batch_size = 64
cache_max_entry_count = 0.5
cache_block_seq_len = 128
quant_policy = 0
use_dynamic_ntk = 0
use_logn_attn = 0
```

## 调节 batch

`max_batch_size`表示推理时最大的 batch 数量

## 调节 k/v cache

k/v cache的内存可通过`cache_block_seq_len`和`cache_max_entry_count`调节。

TurboMind 2.0 实现了 Paged Attention，按块管理 k/v cache。

`cache_block_seq_len` 表示一块 k/v block 可以存放的 token 序列长度，默认 128。TurboMind 按照以下公式计算 k/v block 的内存大小：

```
cache_block_seq_len * num_layer * kv_head_num * size_per_head * 2 * sizeof(kv_data_type))
```

对于 llama2-7b 模型来说，以 half 类型存放 k/v 时，一块 k/v block 的内存为：`128 * 32 * 32 * 128 * 2 * sizeof(half) = 64MB`

`cache_max_entry_count` 根据取值不同，表示不同的含义：

- 当值为 (0, 1) 之间的小数时，`cache_max_entry_count` 表示 k/v block 使用的内存百分比。比如 A100-80G 显卡内存是80G，当`cache_max_entry_count`为0.5时，表示 k/v block 使用的内存为 80 * 0.5 = 40G
- 当值为 > 1的整数时，表示 k/v block 数量

## KV-int8 开关

`quant_policy = 4` 表示打开 KV-int8。使用这个功能时，请先参考 [kv int8](./kv_int8.md) 部署文档导出 KV 量化参数

## 外推能力开关

`use_dynamic_ntk`和`use_logn_attn`和模型的外推能力相关。
