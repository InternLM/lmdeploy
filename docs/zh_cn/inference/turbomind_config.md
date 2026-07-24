# TurboMind 配置

TurboMind 是 LMDeploy 的推理引擎，在用它推理 LLM 模型时，需要把输入模型转成 TurboMind 模型。在 TurboMind 的模型文件夹中，除模型权重外，TurboMind 模型还包括其他一些文件，其中最重要的是和推理性能息息相关的配置文件`triton_models/weights/config.ini`。

如果你使用的是 LMDeploy 0.0.x 版本，请参考[turbomind 1.0 配置](#turbomind-10-配置)章节，了解配置中的相关内容。如果使用的是 LMDeploy 0.1.x 版本，请阅读[turbomind 2.x 配置](#turbomind-2x-配置)了解配置细节。

## TurboMind 2.x 配置

以 `llama-2-7b-chat` 模型为例，在 TurboMind 2.x 中，它的`config.ini`内容如下：

```toml
[llama]
model_name = "llama2"
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
weight_type = "fp16"
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
enable_prefix_caching = false
quant_policy = 0
max_position_embeddings = 2048
rope_scaling_factor = 0.0
use_logn_attn = 0
```

这些参数由模型属性和推理参数组成。模型属性包括层数、head个数、维度等等，它们**不可修改**

```toml
model_name = "llama2"
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

和 TurboMind 1.0 config 相比，TurboMind 2.x config 中的模型属性部分和 1.0 一致，但推理参数发生了变化。

在接下来的章节中，我们重点介绍推理参数。

### 数据类型

和数据类型相关的参数是 `weight_type` 和 `group_size`。它们**不可被修改**。

`weight_type` 表示权重的数据类型。目前支持 fp16 和 int4。int4 表示 4bit 权重。当 `weight_type`为 4bit 权重时，`group_size` 表示 `awq` 量化权重时使用的 group 大小。目前，在 LMDeploy 的预编译包中，使用的是 `group_size = 128`。

### 批处理大小

仍通过 `max_batch_size` 设置最大批处理量。默认值由原来的 32 改成 64。
在 TurboMind 2.x 中，`max_batch_size` 和 `cache_max_entry_count`无关。

### 循环状态数据类型

对于包含 GatedDeltaNet 层的模型，TurboMind 默认使用模型激活值的数据类型保存循环状态。在启动 LMDeploy
前设置 `LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1`，可改用 FP32 保存循环状态。卷积状态仍使用模型激活值的数据
类型。FP32 循环状态可以提高数值精度，但会使用 FP16 或 BF16 两倍的循环状态显存。

```bash
LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1 lmdeploy serve api_server <model-path> --backend turbomind
```

### k/v 缓存大小

`cache_block_seq_len` 和 `cache_max_entry_count` 用来调节 k/v cache 的内存大小。

TurboMind 2.x 实现了 Paged Attention，按块管理 k/v cache。

`cache_block_seq_len` 表示一块 k/v block 可以存放的 token 序列长度，默认 128。TurboMind 按照以下公式计算 k/v block 的内存大小：

```
cache_block_seq_len * num_layer * kv_head_num * size_per_head * 2 * sizeof(kv_data_type)
```

启用序列并行后，该公式仍表示每个 rank 上 block 的实际显存大小。由于 token 交错存储在各个 CP rank 上，调度器中对应的逻辑块覆盖 `cache_block_seq_len * cp` 个全局 token。

对于 llama2-7b 模型来说，以 half 类型存放 k/v 时，一块 k/v block 的内存为：`128 * 32 * 32 * 128 * 2 * sizeof(half) = 64MB`

`cache_max_entry_count` 根据取值不同，表示不同的含义：

- 当值为 (0, 1) 之间的小数时，`cache_max_entry_count` 表示 k/v block 使用的内存百分比。比如 A100-80G 显卡内存是80G，当`cache_max_entry_count`为0.5时，表示 k/v block 使用的内存总量为 80 * 0.5 = 40G
- 当 lmdeploy 版本大于 0.2.1 时，`cache_max_entry_count` 将**空闲**内存的百分比用于 k/v blocks，默认值为 `0.8`。例如，在 A100-80G GPU 上运行 Turbomind 加载 13b 模型时，k/v blocks 使用的内存为 `(80 - 26) * 0.8 = 43.2G`，即利用剩余 54G 中的 80%
- 当值为 > 1的整数时，表示 k/v block 数量

`cache_chunk_size` 表示在每次需要新的 k/v cache 块时，开辟 k/v cache 块的大小。不同的取值，表示不同的含义：

- 当为 > 0 的整数时，开辟 `cache_chunk_size` 个 k/v cache 块
- 当值为 -1 时，开辟 `cache_max_entry_count` 个 k/v cache 块
- 当值为 0 时，时，开辟 `sqrt(cache_max_entry_count)` 个 k/v cache 块

### 前缀缓存开关

`enable_prefix_caching`是前缀缓存（Prefix Caching）功能的开关。值为`True`时表示开启，`False`表示关闭，默认为`False`。

前缀缓存功能主要适用于多个请求具有相同的prompt前缀（比如system prompt）的场景，该相同前缀部分的 k/v block 会被缓存起来，被多个请求重复利用，从而节省了重复计算的开销，提高推理性能。相同prompt前缀长度越长，性能提升越大。

对于前缀缓存的整块复用，一个完整 k/v block 是最小粒度；不开启序列并行时其前缀长度阈值为 `cache_block_seq_len`，开启后为 `cache_block_seq_len * cp`。

### 非整块边界复用

有两个模式开关控制是否在 prompt 与生成边界处发布非整块（partial-block）前缀节点：`cache_prompt`（取 `'all'` 或 `'auto'`，默认 `'auto'`）与 `cache_generation`（取 `'all'`、`'auto'` 或 `'none'`，默认 `'auto'`）。二者都需要开启 `enable_prefix_caching`，且适用于所有开启前缀缓存的模型：所发布的节点携带该非整块的 k/v；对于循环/混合模型（例如包含 GatedDeltaNet 层的模型），该节点还会额外携带循环状态 checkpoint。

将 `cache_prompt='all'` 用于同一 prompt 会被反复处理的场景（多次采样解码、共享/系统 prompt），使重复 prompt 跳过 prefill；默认 `'auto'` 仅对包含图像 token 的非整块 prompt 块生效（复用视觉编码 KV），对纯文本 prompt 无效果。将 `cache_generation='all'` 用于需要从生成精确末端恢复的场景（例如多轮对话）；`'auto'`（默认）仅缓存完整生成块；`'none'` 不缓存任何生成块。非整块节点会带来额外的显存与拷贝带宽开销，除非复用收益明确，否则建议使用 `'auto'`。

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(
    enable_prefix_caching=True,
    cache_prompt='all',
    cache_generation='all',
)
pipe = pipeline('your-model', backend_config=backend_config)
```

- `cache_prompt`：非整块 prompt 边界发布模式，取 `'all'` 或 `'auto'`（默认 `'auto'`）。`'all'` 在 `B = prompt_len - cache_prompt_boundary_skip` 且 `B` 落在块内部时发布可复用的 prompt 边界节点（当 `B` 对齐块边界时还会启用循环状态 checkpoint 钳位），使重复 prompt 跳过 prefill。`'auto'` 仅当该非整块包含图像 token 时生效（复用视觉编码 KV），对纯文本 prompt 无效果。代价是产生该节点的请求需要额外一次 prefill 前向计算，以及（当 `B` 落在块内部时）一个非整块缓存块（partial 块）。
- `cache_prompt_boundary_skip`：将 prompt 末尾的若干 token 视为易变的生成前缀后缀（例如 chat 模板的 `<think>\n`），从可复用的 prompt 边界节点中排除，使节点移动到 `prompt_len - cache_prompt_boundary_skip`。当 `cache_prompt` 为 `'all'` 或 `'auto'` 时生效。默认 1（仅排除最后一个 token）。对于 chat 模板会追加多 token 后缀、且下一轮历史会丢弃该后缀的思考型模型，可调大该值。
- `cache_generation`：生成块缓存模式，取 `'all'`、`'auto'`（默认）或 `'none'`。`'all'` 索引完整生成块及末端非整块，并采用末端循环 frontier checkpoint（精确多轮恢复）。`'auto'` 仅索引完整生成块。`'none'` 不索引任何生成块。当 `'all'` 索引末端非整块时，代价是一个非整块缓存块（partial 块）。无论该设置如何，块边界处的整块 checkpoint 始终会发布。

### kv 量化推理开关

`quant_policy`是 kv 量化和推理开关。

- `quant_policy=4` 代表 4bit k/v 量化和推理
- `quant_policy=8` 代表 8bit k/v 量化和推理

具体使用方法，请参考 [kv quant](../quantization/kv_quant.md) 部署文档

### 外推能力开关

默认 `rope_scaling_factor = 0` 不具备外推能力。设置为 1.0，可以开启 RoPE 的 Dynamic NTK 功能，支持长文本推理。

关于 Dynamic NTK 的原理，详细请参考：

1. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
2. https://kexue.fm/archives/9675

设置 `use_logn_attn = 1`，可以开启 [LogN attention scaling](https://kexue.fm/archives/8823)。

## TurboMind 1.0 配置

以 `llama-2-7b-chat` 模型为例，在 TurboMind 1.0 中，它的`config.ini`内容如下：

```toml
[llama]
model_name = "llama2"
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
weight_type = "fp16"
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

这些参数由模型属性和推理参数组成。模型属性包括层数、head个数、维度等等，它们**不可修改**

```toml
model_name = "llama2"
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

在接下来的章节中，我们重点介绍推理参数。

### 数据类型

和数据类型相关的参数是 `weight_type` 和 `group_size`。它们**不可被修改**。

`weight_type` 表示权重的数据类型。目前支持 fp16 和 int4。int4 表示 4bit 权重。当 `weight_type`为 4bit 权重时，`group_size` 表示 `awq` 量化权重时使用的 group 大小。目前，在 LMDeploy 的预编译包中，使用的是 `group_size = 128`。

### 批处理大小

可通过`max_batch_size`调节推理时最大的 batch 数。一般，batch 越大吞吐量越高。但务必保证 `max_batch_size <= cache_max_entry_count`

### k/v cache 大小

TurboMind 根据 `session_len`、 `cache_chunk_size` 和 `cache_max_entry_count` 开辟 k/v cache 内存。

- `session_len` 表示一个序列的最大长度，即 context window 的大小。
- `cache_chunk_size` 表示当新增对话序列时，每次要开辟多少个序列的 k/v cache
- `cache_max_entry_count` 表示最多缓存多少个对话序列

### kv int8 开关

当启动 8bit k/v 推理时，需要修改参数 `quant_policy` 和 `use_context_fmha`。详细内容请查阅 [kv int8](../quantization/kv_quant.md) 部署文档。

### 外推能力开关

设置 `use_dynamic_ntk = 1`，可以开启 RoPE 的 Dynamic NTK 选项，支持长文本推理。

关于 Dynamic NTK 的原理，详细请参考：

1. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
2. https://kexue.fm/archives/9675

设置 `use_logn_attn = 1`，可以开启 [LogN attention scaling](https://kexue.fm/archives/8823)。
