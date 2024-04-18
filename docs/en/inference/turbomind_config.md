# TurboMind Config

TurboMind is one of the inference engines of LMDeploy. When using it to do model inference, you need to convert the input model into a TurboMind model. In the TurboMind model folder, besides model weight files, the TurboMind model also includes some other files, among which the most important is the configuration file `triton_models/weights/config.ini` that is closely related to inference performance.

If you are using LMDeploy version 0.0.x, please refer to the [turbomind 1.0 config](#turbomind-10-config) section to learn the relevant content in the configuration. Otherwise, please read [turbomind 2.0 config](#turbomind-20-config) to familiarize yourself with the configuration details.

## TurboMind 2.x config

Take the `llama-2-7b-chat` model as an example. In TurboMind 2.0, its config.ini content is as follows:

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
max_batch_size = 64
max_context_token_num = 1
step_length = 1
cache_max_entry_count = 0.5
cache_block_seq_len = 128
cache_chunk_size = 1
quant_policy = 0
max_position_embeddings = 2048
rope_scaling_factor = 0.0
use_logn_attn = 0
```

These parameters are composed of model attributes and inference parameters. Model attributes include the number of layers, the number of heads, dimensions, etc., and they are **not modifiable**.

```toml
model_name = llama2
head_num = 32
kv_head_num = 32
vocab_size = 32000
num_layer = 32
inter_size = 11008
norm_eps = 1e-06
attn_bias = 0
start_id = 1
end_id = 2
rotary_embedding = 128
rope_theta = 10000.0
size_per_head = 128
```

Comparing to TurboMind 1.0, the model attribute part in the config remains the same with TurboMind 1.0, while the inference parameters have changed
In the following sections, we will focus on introducing the inference parameters.

### data type

`weight_type` and `group_size` are the relevant parameters, **which cannot be modified**.

`weight_type` represents the data type of weights. Currently, `fp16` and `int4` are supported. `int4` represents 4bit weights. When `weight_type` is `int4`, `group_size` means the group size used when quantizing weights with `awq`. In LMDeploy prebuilt package, kernels with `group size = 128` are included.

### batch size

The maximum batch size is still set through `max_batch_size`. But its default value has been changed from 32 to 64, and `max_batch_size` is no longer related to `cache_max_entry_count`.

### k/v cache size

k/v cache memory is determined by `cache_block_seq_len` and `cache_max_entry_count`.

TurboMind 2.0 has implemented Paged Attention, managing the k/v cache in blocks.

`cache_block_seq_len` represents the length of the token sequence in a k/v block with a default value 128. TurboMind calculates the memory size of the k/v block according to the following formula:

```
cache_block_seq_len * num_layer * kv_head_num * size_per_head * 2 * sizeof(kv_data_type)
```

For the llama2-7b model, when storing k/v as the `half` type, the memory of a k/v block is: `128 * 32 * 32 * 128 * 2 * sizeof(half) = 64MB`

The meaning of `cache_max_entry_count` varies depending on its value:

- When it's a decimal between (0, 1), `cache_max_entry_count` represents the percentage of memory used by k/v blocks. For example, if turbomind launches on a A100-80G GPU with `cache_max_entry_count` being `0.5`, the total memory used by the k/v blocks is `80 * 0.5 = 40G`.
- When lmdeploy is greater than v0.2.1, `cache_max_entry_count` determines the percentage of **free memory** for k/v blocks, defaulting to `0.8`. For example, with Turbomind on an A100-80G GPU running a 13b model, the memory for k/v blocks would be `(80 - 26) * 0.8 = 43.2G`, utilizing 80% of the free 54G.
- When it's an integer > 0, it represents the total number of k/v blocks

The `cache_chunk_size` indicates the size of the k/v cache chunk to be allocated each time new k/v cache blocks are needed. Different values represent different meanings:

- When it is an integer > 0, `cache_chunk_size` number of k/v cache blocks are allocated.
- When the value is -1, `cache_max_entry_count` number of k/v cache blocks are allocated.
- When the value is 0, `sqrt(cache_max_entry_count)` number of k/v cache blocks are allocated.

### kv quantization and inference switch

- `quant_policy=4` means 4bit k/v quantization and inference
- `quant_policy=8` indicates 8bit k/v quantization and inference

Please refer to [kv quant](../quantization/kv_quant.md) for detailed guide.

### long context switch

By setting `rope_scaling_factor = 1.0`, you can enable the Dynamic NTK option of RoPE, which allows the model to use long-text input and output.

Regarding the principle of Dynamic NTK, please refer to:

1. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
2. https://kexue.fm/archives/9675

You can also turn on [LogN attention scaling](https://kexue.fm/archives/8823) by setting `use_logn_attn = 1`.

## TurboMind 1.0 config

Taking the `llama-2-7b-chat` model as an example, in TurboMind 1.0, its `config.ini` content is as follows:

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
max_batch_size = 32
max_context_token_num = 4
step_length = 1
cache_max_entry_count = 48
cache_chunk_size = 1
use_context_fmha = 1
quant_policy = 0
max_position_embeddings = 2048
use_dynamic_ntk = 0
use_logn_attn = 0
```

These parameters are composed of model attributes and inference parameters. Model attributes include the number of layers, the number of heads, dimensions, etc., and they are **not modifiable**.

```toml
model_name = llama2
head_num = 32
kv_head_num = 32
vocab_size = 32000
num_layer = 32
inter_size = 11008
norm_eps = 1e-06
attn_bias = 0
start_id = 1
end_id = 2
rotary_embedding = 128
rope_theta = 10000.0
size_per_head = 128
```

In the following sections, we will focus on introducing the inference parameters.

### data type

`weight_type` and `group_size` are the relevant parameters, **which cannot be modified**.

`weight_type` represents the data type of weights. Currently, `fp16` and `int4` are supported. `int4` represents 4bit weights. When `weight_type` is `int4`, `group_size` means the group size used when quantizing weights with `awq`. In LMDeploy prebuilt package, kernels with `group size = 128` are included.

### batch size

`max_batch_size` determines the max size of a batch during inference. In general, the larger the batch size is, the higher the throughput is. But make sure that `max_batch_size <= cache_max_entry_count`

### k/v cache size

TurboMind allocates k/v cache memory based on `session_len`, `cache_chunk_size`, and `cache_max_entry_count`.

- `session_len` denotes the maximum length of a sequence, i.e., the size of the context window.
- `cache_chunk_size` indicates the size of k/v sequences to be allocated when new sequences are added.
- `cache_max_entry_count` signifies the maximum number of k/v sequences that can be cached.

### kv int8 switch

When initiating 8bit k/v inference, change `quant_policy = 4` and `use_context_fmha = 0`. Please refer to [kv int8](../quantization/kv_quant.md) for a guide.

### long context switch

By setting `use_dynamic_ntk = 1`, you can enable the Dynamic NTK option of RoPE, which allows the model to use long-text input and output.

Regarding the principle of Dynamic NTK, please refer to:

1. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
2. https://kexue.fm/archives/9675

You can also turn on [LogN attention scaling](https://kexue.fm/archives/8823) by setting `use_logn_attn = 1`.
