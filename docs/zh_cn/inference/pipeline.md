# 推理 pipeline

本文首先通过一些例子展示 pipeline 的基本用法。然后详细介绍 pipeline API 的参数以及参数的具体设置。

## 使用方法

使用默认参数的例子:

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

展示如何设置 tp 数的例子:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
pipe = pipeline('internlm/internlm-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

展示如何设置 sampling 参数:

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

展示如何设置 OpenAI 格式输入的例子:

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm-chat-7b',
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

展示 pytorch 后端的例子,需要先安装 triton:

```shell
pip install triton>=2.1.0
```

```python
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

backend_config = PytorchEngineConfig(session_len=2024)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm-chat-7b',
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

## `pipeline` API

`pipeline`函数是一个更高级别的 API，设计用于让用户轻松实例化和使用 AsyncEngine。

### 初始化参数:

| Parameter            | Type                                                 | Description                                                                                              | Default                     |
| -------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------- |
| model_path           | str                                                  | 模型路径。这可以是存储 Turbomind 模型的本地目录的路径，或者是托管在 huggingface.co 上的模型的 model_id。 |                             |
| model_name           | Optional\[str\]                                      | 当 model_path 指向 huggingface.co 上的 Pytorch 模型时需要的模型名称。                                    | None                        |
| backend_config       | TurbomindEngineConfig \| PytorchEngineConfig \| None | 后端的配置对象。根据所选后端，可以是 TurbomindEngineConfig 或 PytorchEngineConfig。                      | None, 默认跑 turbomind 后端 |
| chat_template_config | Optional\[ChatTemplateConfig\]                       | 聊天模板的配置。                                                                                         | None                        |
| log_level            | str                                                  | 日志级别。                                                                                               | 'ERROR'                     |

### 调用

| 参数名称           | 数据类型                   | 默认值 | 描述                                                                                                                                      |
| ------------------ | -------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| prompts            | List\[str\]                | None   | 批处理的提示信息。                                                                                                                        |
| gen_config         | GenerationConfig 或者 None | None   | GenerationConfig 的一个实例。默认为None。                                                                                                 |
| do_preprocess      | bool                       | True   | 是否预处理消息。默认为True，表示将应用chat_template。                                                                                     |
| request_output_len | int                        | 512    | 输出的token数。后期会弃用，请改用 gen_config 参数                                                                                         |
| top_k              | int                        | 40     | 在进行top-k过滤时，保留概率最高的词汇token的数量。后期会弃用，请改用 gen_config 参数                                                      |
| top_p              | float                      | 0.8    | 如果设置为小于1的浮点数，则只有那些最有可能的token集合（其概率累加到top_p或更高）才会被保留以用于生成。后期会弃用，请改用 gen_config 参数 |
| temperature        | float                      | 0.8    | 用于调节下一个token的概率。后期会弃用，请改用 gen_config 参数                                                                             |
| repetition_penalty | float                      | 1.0    | 重复惩罚的参数。1.0表示没有惩罚。后期会弃用，请改用 gen_config 参数                                                                       |
| ignore_eos         | bool                       | False  | 是否忽略结束符的指示器。后期会弃用，请改用 gen_config 参数                                                                                |

## TurbomindEngineConfig

### 描述

这个类提供了TurboMind引擎的配置参数。

### 参数

| Parameter             | Type          | Description                                                            | Default |
| --------------------- | ------------- | ---------------------------------------------------------------------- | ------- |
| model_format          | str, optional | 已部署模型的布局。可以是以下值之一：`hf`, `llama`, `awq`。             | None    |
| tp                    | int           | 在张量并行中使用的GPU卡数量。                                          | 1       |
| session_len           | int, optional | 序列的最大会话长度。                                                   | None    |
| max_batch_size        | int           | 推理过程中的最大批处理大小。                                           | 128     |
| cache_max_entry_count | float         | 由k/v缓存占用的GPU内存百分比。                                         | 0.5     |
| quant_policy          | int           | 默认为0。当k/v量化为8位时，设置为4。                                   | 0       |
| rope_scaling_factor   | float         | 用于动态ntk的缩放因子。TurboMind遵循transformer LlamaAttention的实现。 | 0.0     |
| use_logn_attn         | bool          | 是否使用对数注意力。                                                   | False   |

## PytorchEngineConfig

### 描述

此类是PyTorch引擎的配置对象。

### 参数

| Parameter        | Type | Description                                                  | Default     |
| ---------------- | ---- | ------------------------------------------------------------ | ----------- |
| model_name       | str  | 已部署模型的对话模板名称。                                   | ''          |
| tp               | int  | 张量并行度。                                                 | 1           |
| session_len      | int  | 最大会话长度。                                               | None        |
| max_batch_size   | int  | 最大批处理大小。                                             | 128         |
| eviction_type    | str  | 当kv缓存满时需要执行的操作，可选值为\['recompute', 'copy'\]. | 'recompute' |
| prefill_interval | int  | 执行预填充的间隔。                                           | 16          |
| block_size       | int  | 分页缓存块大小。                                             | 64          |
| num_cpu_blocks   | int  | CPU块的数量。如果值为0，缓存将根据当前环境进行分配。         | 0           |
| num_gpu_blocks   | int  | GPU块的数量。如果值为0，缓存将根据当前环境进行分配。         | 0           |
| adapters         | dict | lora adapters的配置路径                                      | None        |

## GenerationConfig

### 描述

这个类包含了由推理引擎使用的生成参数。

### 参数

| Parameter          | Type        | Description                                           | Default |
| ------------------ | ----------- | ----------------------------------------------------- | ------- |
| n                  | int         | 对每个输入消息生成聊天补全选择的数量。目前仅支持 1    | 1       |
| max_new_tokens     | int         | 聊天补全中可以生成的最大令牌数。                      | 512     |
| top_p              | float       | 核心采样，其中模型考虑具有top_p概率质量的令牌。       | 1.0     |
| top_k              | int         | 模型考虑具有最高概率的前K个令牌。                     | 1       |
| temperature        | float       | 采样温度。                                            | 0.8     |
| repetition_penalty | float       | 防止模型生成重复词或短语的惩罚。大于1的值会抑制重复。 | 1.0     |
| ignore_eos         | bool        | 是否忽略eos_token_id。                                | False   |
| random_seed        | int         | 采样令牌时使用的种子。                                | None    |
| stop_words         | List\[str\] | 停止进一步生成令牌的词。                              | None    |
| bad_words          | List\[str\] | 引擎永远不会生成的词。                                | None    |

## FAQs

- *RuntimeError: context has already been set*. 如果你在使用 tp>1 和 pytorch 后端的时候，遇到了这个错误。请确保 python 脚本中有下面内容作为入口
  ```python
  if __name__ == '__main__':
  ```
  一般来说，在多线程或多进程上下文中，可能需要确保初始化代码只执行一次。这时候，`if __name__ == '__main__':` 可以帮助确保这些初始化代码只在主程序执行，而不会在每个新创建的进程或线程中重复执行。
