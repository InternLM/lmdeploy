# Reasoning Outputs

对于支持推理能力的模型，比如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，LMDeploy 支持在服务端解析推理结果，并通过 `reasoning_content` 单独返回推理内容。

## 使用示例

### DeepSeek R1

我们可以像启动其他模型一样启动 DeepSeek R1 的 `api_server`，但需要额外指定 `--reasoning-parser` 参数。

```
lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek-r1
```

然后，我们就可以在客户端调用这个服务的功能：

```python
from openai import OpenAI

openai_api_key = "Your API key"
openai_api_base = "http://0.0.0.0:23333/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(model=model, messages=messages, stream=True)
for stream_response in response:
    print('reasoning content: ',stream_response.choices[0].delta.reasoning_content)
    print('content: ', stream_response.choices[0].delta.content)

response = client.chat.completions.create(model=model, messages=messages, stream=False)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content:", reasoning_content)
print("content:", content)
```

## 自定义 parser

内置的 reasoning parser 名称包括：

- `qwen-qwq`
- `qwen3`
- `intern-s1`
- `deepseek-r1`
- `deepseek-v3`
- `gpt-oss`

### 说明

- `deepseek-v3`：仅当 `enable_thinking=True` 时，才会从推理模式开始解析。
  当 `enable_thinking` 为 `None`（默认）时，通常不会出现推理段，输出为普通内容。
- `gpt-oss`：基于 OpenAI Harmony channel 解析：
  - `final` -> `content`
  - `analysis` -> `reasoning_content`
  - `commentary` 且 `recipient` 为 `functions.*` -> `tool_calls`

### 添加自定义 parser

在 `lmdeploy/serve/openai/reasoning_parser/` 目录下新增 parser 类，并通过 `ReasoningParserManager` 注册。

```python
from lmdeploy.serve.openai.reasoning_parser import (
    ReasoningParser, ReasoningParserManager
)

@ReasoningParserManager.register_module(["example"])
class ExampleParser(ReasoningParser):
    def __init__(self, tokenizer: object, **kwargs):
        super().__init__(tokenizer, **kwargs)

    def get_reasoning_open_tag(self) -> str | None:
        return "<think>"

    def get_reasoning_close_tag(self) -> str | None:
        return "</think>"

    def starts_in_reasoning_mode(self) -> bool:
        return True
```

然后通过以下命令启动服务：

```
lmdeploy serve api_server $model_path --reasoning-parser example
```
