## Pipeline

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

- **stop_session**(session_id): 停止具有给定 id 的会话。

- **end_session**(session_id): 清除具有给定 id 的会话。

- **get_generator**(stop, session_id): 获取给定会话的生成器。

- **batch_infer**(prompts, gen_config, chat_template_config, request_output_len, top_k, top_p, temperature, repetition_penalty, ignore_eos, kwargs): 对一批提示进行推理。

- **generate**(messages, session_id, gen_config, chat_template_config, stream_response, sequence_start, sequence_end, step, request_output_len, stop, stop_words, top_k, top_p, temperature, repetition_penalty, ignore_eos, kwargs): 生成响应。

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

```python
import lmdeploy

# Initialize pipeline
pipe = lmdeploy.pipeline('InternLM/internlm-chat-7b-v1_1')

# Perform inference on multiple inputs
response = pipe(['hi','say this is a test'])

print(response)

```

在上述示例中，pipeline 函数初始化了一个 AsyncEngine，该引擎使用通过 'model_path' 指定的模型。然后，它使用引擎的 `__call__` 方法对提示列表进行推理，并打印出由模型生成的响应。
