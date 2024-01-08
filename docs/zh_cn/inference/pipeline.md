## Pipeline

### `pipeline` API

`pipeline`函数是一个更高级别的 API，设计用于让用户轻松实例化和使用 AsyncEngine。

#### 参数:

- **model_path** (str): 模型路径。这可以是存储 Turbomind 模型的本地目录的路径，或者是托管在 (huggingface.co)\[https://huggingface.co\] 上的模型的 model_id。
- **model_name** (Optional\[str\]): 当 model_path 指向 huggingface.co 上的 Pytorch 模型时需要的模型名称。默认为 None。
- **backend** (Literal\['turbomind', 'pytorch'\]): 指定要使用的后端，可选 turbomind 或 pytorch。默认设置为 turbomind。
- **backend_config** (Optional\[Union\[TurbomindEngineConfig, PytorchEngineConfig\]\]): 后端的配置对象。根据所选后端，可以是 TurbomindEngineConfig 或 PytorchEngineConfig。默认为 None。
- **chat_template_config** (Optional\[ChatTemplateConfig\]): 聊天模板的配置。默认为 None。
- **instance_num** (int): 处理并发请求时要创建的实例数。默认为 32。
- **tp** (int): 张量并行单位的数量。默认为 1。
- **log_level** (str): 日志级别。默认为 'ERROR'。

### 示例

pytorch 后端的例子:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.pytorch import EngineConfig

# there are more arguments in EngineConfig set by default
backend_config = EngineConfig(tp = 1, session_len= 1024)

# there are more arguments in GenerationConfig set by default
gen_config = GenerationConfig(max_new_tokens=224)

# Initialize pipeline
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',  backend='pytorch', backend_config = backend_config)

# Perform inference on multiple inputs
response = pipe(['hi','say this is a test'], gen_config=gen_config)

print(response)

```

turbomind 后端的例子:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.turbomind import EngineConfig

# there are more arguments in EngineConfig set by default
backend_config = EngineConfig(tp = 1, session_len= 1024)

# there are more arguments in GenerationConfig set by default
gen_config = GenerationConfig(max_new_tokens=224)

# Initialize pipeline
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',  backend='turbomind', backend_config = backend_config)

# Perform inference on multiple inputs
response = pipe(['hi','say this is a test'], gen_config=gen_config)

print(response)

```

在上述示例中，pipeline 函数初始化了一个 AsyncEngine，该引擎使用通过 'model_path' 指定的模型。然后，它使用引擎的 `__call__` 方法对提示列表进行推理，并打印出由模型生成的响应。

### `AsyncEngine` API

`AsyncEngine` 类提供了一个异步推理引擎，该引擎维护了多个指定模型的实例以实现高效处理。

#### 初始化参数:

- **model_path** (str): 模型路径。这可以是存储 Turbomind 模型的本地目录的路径，或者是托管在 (huggingface.co)\[https://huggingface.co\] 上的模型的 model_id。
- **model_name** (Optional\[str\]): 当 model_path 指向 huggingface.co 上的 Pytorch 模型时需要的模型名称。默认为 None。
- **backend** (Literal\['turbomind', 'pytorch'\]): 指定要使用的后端，可选 turbomind 或 pytorch。默认设置为 turbomind。
- **backend_config** (Optional\[Union\[TurbomindEngineConfig, PytorchEngineConfig\]\]): 后端的配置对象。根据所选后端，可以是 TurbomindEngineConfig 或 PytorchEngineConfig。默认为 None。
- **chat_template_config** (Optional\[ChatTemplateConfig\]): 聊天模板的配置。默认为 None。
- **instance_num** (int): 处理并发请求时要创建的实例数。默认为 32。
- **tp** (int): 张量并行单位的数量。默认为 1。

**Methods**:

- **call**(prompts, gen_config, chat_template_config, request_output_len, top_k, top_p, temperature, repetition_penalty, ignore_eos, kwargs): 对一批提示进行推理。

- **stop_session**(session_id): 停止一个正在响应的具有给定 id 的会话。

- **end_session**(session_id): 清除具有给定 id 的会话。

- **get_generator**(stop, session_id): 获取给定会话的生成器。

- **batch_infer**(prompts, gen_config, chat_template_config, request_output_len, top_k, top_p, temperature, repetition_penalty, ignore_eos, kwargs): 对一批提示进行推理。

- **generate**(messages, session_id, gen_config, chat_template_config, stream_response, sequence_start, sequence_end, step, request_output_len, stop, stop_words, top_k, top_p, temperature, repetition_penalty, ignore_eos, kwargs): 生成响应。

### EngineConfig(pytorch)

#### 描述

此类是PyTorch引擎的配置对象。

#### 参数

- **model_name** (str): 指定模型的名称。默认值为空字符串。
- **tp** (int): 张量并行度。默认为1。
- **session_len** (int): 最大会话长度。默认为None。
- **max_batch_size** (int): 最大批处理大小。默认为128。
- **eviction_type** (str): 当kv缓存满时需要执行的操作，可选值为\['recompute', 'copy'\]。默认为'recompute'。
- **prefill_interval** (int): 执行预填充的间隔。默认为16。
- **block_size** (int): 分页缓存块大小。默认为64。
- **num_cpu_blocks** (int): CPU块的数量。如果值为0，缓存将根据当前环境进行分配。默认为0。
- **num_gpu_blocks** (int): GPU块的数量。如果值为0，缓存将根据当前环境进行分配。默认为0。

### EngineConfig (turbomind)

#### 描述

这个类提供了TurboMind引擎的配置参数。

#### 参数

- **model_name** (str, 可选): 已部署模型的名称。默认为None。
- **model_format** (str, 可选): 已部署模型的布局。可以是以下值之一：`hf`, `llama`, `awq`。默认为None。
- **group_size** (int): 在将权重量化为4位时使用的组大小。默认为128。
- **tp** (int): 在张量并行中使用的GPU卡数量。默认为1。
- **session_len** (int, 可选): 序列的最大会话长度。默认为None。
- **max_batch_size** (int): 推理过程中的最大批处理大小。默认为128。
- **max_context_token_num** (int): 每次前向传播中需要处理的最大令牌数量。默认为1。
- **cache_max_entry_count** (float): 由k/v缓存占用的GPU内存百分比。默认为0.5。
- **cache_block_seq_len** (int): k/v块中的序列长度。默认为128。
- **cache_chunk_size** (int): TurboMind引擎试图从GPU内存重新分配的每次的块数。默认为-1。
- **num_tokens_per_iter** (int): 每次迭代处理的令牌数。默认为0。
- **max_prefill_iters** (int): 单个请求的最大预填充迭代次数。默认为1。
- **use_context_fmha** (int): 是否在上下文解码中使用fmha。默认为1。
- **quant_policy** (int): 默认为0。当k/v量化为8位时，设置为4。
- **rope_scaling_factor** (float): 用于动态ntk的缩放因子。TurboMind遵循transformer LlamaAttention的实现。默认为0.0。
- **use_dynamic_ntk** (bool): 是否使用动态ntk。默认为False。
- **use_logn_attn** (bool): 是否使用对数注意力。默认为False。
- **kv_bits** (int): 量化后的k/v的位数。默认为8。

### GenerationConfig

#### 描述

这个类包含了由推理引擎使用的生成参数。

#### 参数

- **n** (int): 对每个输入消息生成聊天补全选择的数量。默认为1。
- **max_new_tokens** (int): 聊天补全中可以生成的最大令牌数。默认为512。
- **top_p** (float): 核心采样，其中模型考虑具有top_p概率质量的令牌。默认为1.0。
- **top_k** (int): 模型考虑具有最高概率的前K个令牌。默认为1。
- **temperature** (float): 采样温度。默认为0.8。
- **repetition_penalty** (float): 防止模型生成重复词或短语的惩罚。大于1的值会抑制重复。 默认为1.0。
- **ignore_eos** (bool): 是否忽略eos_token_id。默认为False。
- **random_seed** (int): 采样令牌时使用的种子。默认为None。
- **stop_words** (List\[str\]): 停止进一步生成令牌的词。默认为None。
- **bad_words** (List\[str\]): 引擎永远不会生成的词。默认为None。
