# 推理 pipeline

本文首先通过一些例子展示 pipeline 的基本用法。然后详细介绍 pipeline API 的参数以及参数的具体设置。

## 使用方法

- **使用默认参数的例子:**

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm2-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

在这个例子中，pipeline 默认申请一定比例显存，用来存储推理过程中产生的 k/v。比例由参数 `TurbomindEngineConfig.cache_max_entry_count` 控制。

LMDeploy 在研发过程中，k/v cache 比例的设定策略有变更，以下为变更记录：

1. `v0.2.0 <= lmdeploy <= v0.2.1`

   默认比例为 0.5，表示 **GPU总显存**的 50% 被分配给 k/v cache。 对于 7B 模型来说，如果显存小于 40G，会出现 OOM。当遇到 OOM 时，请按照下面的方法，酌情降低 k/v cache 占比：

   ```python
   from lmdeploy import pipeline, TurbomindEngineConfig

   # 调低 k/v cache内存占比调整为总显存的 20%
   backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

   pipe = pipeline('internlm/internlm2-chat-7b',
                   backend_config=backend_config)
   response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
   print(response)
   ```

2. `lmdeploy > v0.2.1`

   分配策略改为从**空闲显存**中按比例为 k/v cache 开辟空间。默认比例值调整为 0.8。如果遇到 OOM，类似上面的方法，请酌情减少比例值，降低 k/v cache 的内存占用量

- **如何设置 tp:**

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

- **如何设置 sampling 参数:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

- **如何设置 OpenAI 格式输入:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
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

- **流式返回处理结果：**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
for item in pipe.stream_infer(prompts, gen_config=gen_config):
    print(item)
```

- **使用 pytorch 后端**

需要先安装 triton

```shell
pip install triton>=2.1.0
```

```python
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

backend_config = PytorchEngineConfig(session_len=2048)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
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

### Response

| 参数名             | 类型                                    | 描述                                                                                                                                         |
| ------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| text               | str                                     | 服务器响应的文本。如果输出文本为空字符串，且finish_reason为length，则表示已达最大会话长度。                                                  |
| generate_token_len | int                                     | 响应的 token 数。                                                                                                                            |
| input_token_len    | int                                     | 输入提示 token 数。注意，这可能包含聊天模板部分。                                                                                            |
| session_id         | int                                     | 运行会话的ID。基本上，它指的是输入请求批次的索引位置。                                                                                       |
| finish_reason      | Optional\[Literal\['stop', 'length'\]\] | 模型停止生成 token 的原因。如果模型遇到 stop word，这将设置为'stop'；如果达到了请求中指定的最大 token 数或者 session_len，则设置为'length'。 |

## TurbomindEngineConfig

### 描述

这个类提供了TurboMind引擎的配置参数。

### 参数

| Parameter             | Type          | Description                                                            | Default |
| --------------------- | ------------- | ---------------------------------------------------------------------- | ------- |
| model_name            | str, optional | 已部署模型的对话模板名称。在版本 > 0.2.1 之后已废弃                    | None    |
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

| Parameter           | Type        | Description                                           | Default |
| ------------------- | ----------- | ----------------------------------------------------- | ------- |
| n                   | int         | 对每个输入消息生成聊天补全选择的数量。目前仅支持 1    | 1       |
| max_new_tokens      | int         | 聊天补全中可以生成的最大令牌数。                      | 512     |
| top_p               | float       | 核心采样，其中模型考虑具有top_p概率质量的令牌。       | 1.0     |
| top_k               | int         | 模型考虑具有最高概率的前K个令牌。                     | 1       |
| temperature         | float       | 采样温度。                                            | 0.8     |
| repetition_penalty  | float       | 防止模型生成重复词或短语的惩罚。大于1的值会抑制重复。 | 1.0     |
| ignore_eos          | bool        | 是否忽略eos_token_id。                                | False   |
| random_seed         | int         | 采样令牌时使用的种子。                                | None    |
| stop_words          | List\[str\] | 停止进一步生成令牌的词。                              | None    |
| bad_words           | List\[str\] | 引擎永远不会生成的词。                                | None    |
| min_new_tokens      | int         | 最小令牌生成数。                                      | None    |
| skip_special_tokens | bool        | 是否跳过 special token。                              | True    |

## FAQs

- *RuntimeError: context has already been set*. 如果你在使用 tp>1 和 pytorch 后端的时候，遇到了这个错误。请确保 python 脚本中有下面内容作为入口
  ```python
  if __name__ == '__main__':
  ```
  一般来说，在多线程或多进程上下文中，可能需要确保初始化代码只执行一次。这时候，`if __name__ == '__main__':` 可以帮助确保这些初始化代码只在主程序执行，而不会在每个新创建的进程或线程中重复执行。
