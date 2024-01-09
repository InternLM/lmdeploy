## Pipeline

### `pipeline` API

`pipeline`函数是一个更高级别的 API，设计用于让用户轻松实例化和使用 AsyncEngine。

#### 参数:

| Parameter            | Type                                                 | Description                                                                                              | Default     |
| -------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------- |
| model_path           | str                                                  | 模型路径。这可以是存储 Turbomind 模型的本地目录的路径，或者是托管在 huggingface.co 上的模型的 model_id。 |             |
| model_name           | Optional\[str\]                                      | 当 model_path 指向 huggingface.co 上的 Pytorch 模型时需要的模型名称。                                    | None        |
| backend              | Literal\['turbomind', 'pytorch'\]                    | 指定要使用的后端，可选 turbomind 或 pytorch。                                                            | 'turbomind' |
| backend_config       | TurbomindEngineConfig \| PytorchEngineConfig \| None | 后端的配置对象。根据所选后端，可以是 TurbomindEngineConfig 或 PytorchEngineConfig。                      | None        |
| chat_template_config | Optional\[ChatTemplateConfig\]                       | 聊天模板的配置。                                                                                         | None        |
| instance_num         | int                                                  | 处理并发请求时要创建的实例数。                                                                           | 32          |
| tp                   | int                                                  | 张量并行单位的数量。                                                                                     | 1           |
| log_level            | str                                                  | 日志级别。                                                                                               | 'ERROR'     |

### 示例

使用默认参数的例子:

```python
import lmdeploy

pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

展示如何设置 tp 数的例子:

```python
import lmdeploy
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

展示如何设置 sampling 参数:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

展示如何设置 OpenAI 格式输入的例子:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts,
                gen_config=gen_config)
print(response)
```

展示 pytorch 后端的例子:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.pytorch import EngineConfig

backend_config = EngineConfig(session_len=2048)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend='pytorch',
                         backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts, gen_config=gen_config)
print(response)
```

### EngineConfig(pytorch)

#### 描述

此类是PyTorch引擎的配置对象。

#### 参数

| Parameter        | Type | Description                                                  | Default     |
| ---------------- | ---- | ------------------------------------------------------------ | ----------- |
| model_name       | str  | 指定模型的名称。                                             | ''          |
| tp               | int  | 张量并行度。                                                 | 1           |
| session_len      | int  | 最大会话长度。                                               | None        |
| max_batch_size   | int  | 最大批处理大小。                                             | 128         |
| eviction_type    | str  | 当kv缓存满时需要执行的操作，可选值为\['recompute', 'copy'\]. | 'recompute' |
| prefill_interval | int  | 执行预填充的间隔。                                           | 16          |
| block_size       | int  | 分页缓存块大小。                                             | 64          |
| num_cpu_blocks   | int  | CPU块的数量。如果值为0，缓存将根据当前环境进行分配。         | 0           |
| num_gpu_blocks   | int  | GPU块的数量。如果值为0，缓存将根据当前环境进行分配。         | 0           |

### EngineConfig (turbomind)

#### 描述

这个类提供了TurboMind引擎的配置参数。

#### 参数

| Parameter             | Type          | Description                                                            | Default |
| --------------------- | ------------- | ---------------------------------------------------------------------- | ------- |
| model_name            | str, optional | 已部署模型的名称。                                                     | None    |
| model_format          | str, optional | 已部署模型的布局。可以是以下值之一：`hf`, `llama`, `awq`。             | None    |
| group_size            | int           | 在将权重量化为4位时使用的组大小。                                      | 0       |
| tp                    | int           | 在张量并行中使用的GPU卡数量。                                          | 1       |
| session_len           | int, optional | 序列的最大会话长度。                                                   | None    |
| max_batch_size        | int           | 推理过程中的最大批处理大小。                                           | 128     |
| max_context_token_num | int           | 每次前向传播中需要处理的最大令牌数量。                                 | 1       |
| cache_max_entry_count | float         | 由k/v缓存占用的GPU内存百分比。                                         | 0.5     |
| cache_block_seq_len   | int           | k/v块中的序列长度。                                                    | 128     |
| cache_chunk_size      | int           | TurboMind引擎试图从GPU内存重新分配的每次的块数。                       | -1      |
| num_tokens_per_iter   | int           | 每次迭代处理的令牌数。                                                 | 0       |
| max_prefill_iters     | int           | 单个请求的最大预填充迭代次数。                                         | 1       |
| use_context_fmha      | int           | 是否在上下文解码中使用fmha。                                           | 1       |
| quant_policy          | int           | 默认为0。当k/v量化为8位时，设置为4。                                   | 0       |
| rope_scaling_factor   | float         | 用于动态ntk的缩放因子。TurboMind遵循transformer LlamaAttention的实现。 | 0.0     |
| use_dynamic_ntk       | bool          | 是否使用动态ntk。                                                      | False   |
| use_logn_attn         | bool          | 是否使用对数注意力。                                                   | False   |

### GenerationConfig

#### 描述

这个类包含了由推理引擎使用的生成参数。

#### 参数

| Parameter          | Type        | Description                                           | Default |
| ------------------ | ----------- | ----------------------------------------------------- | ------- |
| n                  | int         | 对每个输入消息生成聊天补全选择的数量。                | 1       |
| max_new_tokens     | int         | 聊天补全中可以生成的最大令牌数。                      | 512     |
| top_p              | float       | 核心采样，其中模型考虑具有top_p概率质量的令牌。       | 1.0     |
| top_k              | int         | 模型考虑具有最高概率的前K个令牌。                     | 1       |
| temperature        | float       | 采样温度。                                            | 0.8     |
| repetition_penalty | float       | 防止模型生成重复词或短语的惩罚。大于1的值会抑制重复。 | 1.0     |
| ignore_eos         | bool        | 是否忽略eos_token_id。                                | False   |
| random_seed        | int         | 采样令牌时使用的种子。                                | None    |
| stop_words         | List\[str\] | 停止进一步生成令牌的词。                              | None    |
| bad_words          | List\[str\] | 引擎永远不会生成的词。                                | None    |
