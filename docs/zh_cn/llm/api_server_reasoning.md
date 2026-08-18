# Reasoning Outputs

对于支持推理能力的模型，比如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，LMDeploy 支持在服务端解析推理结果，并通过 `reasoning_content` 单独返回推理内容。

## 使用示例

### DeepSeek R1

我们可以像启动其他模型一样启动 DeepSeek R1 的 `api_server`，但需要额外指定 `--reasoning-parser` 参数。

```
lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser default
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

### 在多轮对话中保留推理内容

部分模型可以复用先前 assistant 轮次的推理内容。对于 Qwen3.8，请通过
`chat_template_kwargs` 传入 `preserve_thinking`，并在构造下一次请求时保留
assistant 的 `reasoning_content`：

```python
messages.append({
    "role": "assistant",
    "reasoning_content": reasoning_content,
    "content": content,
})
messages.append({"role": "user", "content": "请换一种方式解释这个结果。"})

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"preserve_thinking": True}},
)
```

不传 `preserve_thinking` 时，由模型的聊天模板决定默认行为；Qwen3.8 默认保留历史
推理。将其设为 `False` 可移除已经完成的较早轮次中的推理。保留的推理内容会
成为输入提示词的一部分，并增加输入 Token 数量。

## 自定义 parser

`--reasoning-parser` 当前支持以下注册名称：

- `default`：使用常见的 `<think>...</think>` 协议，适用于 Qwen3、QwQ、
  DeepSeek R1、Intern-S1 以及兼容模型。
- `deepseek-v3`：仅当 `enable_thinking=True` 时从推理模式开始解析。
- `deepseek-v32` 和 `deepseek-v3.2`：DeepSeek V3.2 的两个别名；当
  `thinking=True` 或 `enable_thinking=True` 时从推理模式开始解析。
- `deepseek-v4`：使用与 DeepSeek V3.2 parser 相同的模式开关。

旧名称 `qwen-qwq`、`intern-s1` 和 `deepseek-r1` 仍会映射到
`default`，但会产生弃用警告。GPT-OSS 不使用注册的 reasoning parser；
LMDeploy 会自动为它选择专用的 OpenAI Harmony response parser。

### 添加自定义 parser

在 `lmdeploy/serve/parsers/reasoning_parser/` 下创建模块，并通过
`ReasoningParserManager` 注册。Reasoning parser 负责声明协议的起止标签，
以及生成是否从推理模式开始；统一的 response parser 会处理流式和非流式响应
的内容拆分。

```python
# lmdeploy/serve/parsers/reasoning_parser/example_reasoning_parser.py
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name="example")
class ExampleReasoningParser(ReasoningParser):
    """解析使用 <reasoning>...</reasoning> 协议的模型。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_thinking = kwargs.get("enable_thinking")

    @classmethod
    def get_reasoning_open_tag(cls) -> str:
        return "<reasoning>"

    @classmethod
    def get_reasoning_close_tag(cls) -> str:
        return "</reasoning>"

    def starts_in_reasoning_mode(self) -> bool:
        return self.enable_thinking is not False
```

还需要在 `lmdeploy/serve/parsers/reasoning_parser/__init__.py` 中导入该模块，
确保 CLI 校验参数之前完成注册：

```python
from .example_reasoning_parser import ExampleReasoningParser  # noqa: F401
```

服务启动时，LMDeploy 会检查上述起止标签是否是模型 tokenizer 中的独立
Token。请将示例标签替换成模型实际使用的协议 Token；如果模型需要不同的校验
逻辑，请重写 `validate_tokenizer`。

然后启动服务：

```bash
lmdeploy serve api_server "$MODEL_PATH" --reasoning-parser example
```
