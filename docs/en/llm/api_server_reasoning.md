# Reasoning Outputs

For models that support reasoning capabilities, such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), LMDeploy can parse reasoning outputs on the server side and expose them via `reasoning_content`.

## Examples

### DeepSeek R1

We can start DeepSeek R1's `api_server` like other models, but we need to specify the `--reasoning-parser` argument.

```
lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek-r1
```

Then, we can call the service's functionality from the client:

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

## Custom parser

Built-in reasoning parser names include:

- `qwen-qwq`
- `qwen3`
- `intern-s1`
- `deepseek-r1`
- `deepseek-v3`
- `gpt-oss`

### Notes

- `deepseek-v3`: starts in reasoning mode only when `enable_thinking=True`.
  When `enable_thinking` is `None` (default), output is usually plain content without a reasoning segment.
- `gpt-oss`: parses OpenAI Harmony channels:
  - `final` -> `content`
  - `analysis` -> `reasoning_content`
  - `commentary` with `functions.*` recipient -> `tool_calls`

### Add a custom parser

Add a parser class under `lmdeploy/serve/openai/reasoning_parser/` and register it with `ReasoningParserManager`.

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

Then start the service with:

```
lmdeploy serve api_server $model_path --reasoning-parser example
```
