# Reasoning Outputs

For models that support reasoning capabilities, such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), LMDeploy can parse reasoning outputs on the server side and expose them via `reasoning_content`.

## Examples

### DeepSeek R1

We can start DeepSeek R1's `api_server` like other models, but we need to specify the `--reasoning-parser` argument.

```
lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser default
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

### Preserve reasoning in multi-turn conversations

Some models can reuse reasoning from earlier assistant turns. For Qwen3.8,
pass `preserve_thinking` through `chat_template_kwargs` and include the
assistant's `reasoning_content` when building the next request:

```python
messages.append({
    "role": "assistant",
    "reasoning_content": reasoning_content,
    "content": content,
})
messages.append({"role": "user", "content": "Explain that result another way."})

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"preserve_thinking": True}},
)
```

Omitting `preserve_thinking` leaves the behavior to the
model's chat template; Qwen3.8 preserves earlier reasoning by default. Set it
to `False` to remove reasoning from completed earlier turns. Preserved
reasoning is part of the input prompt and increases its token count.

## Custom parser

The registered `--reasoning-parser` names are:

- `default`: the common `<think>...</think>` protocol used by Qwen3,
  QwQ, DeepSeek R1, Intern-S1, and compatible models.
- `deepseek-v3`: starts in reasoning mode only when
  `enable_thinking=True`.
- `deepseek-v32` and `deepseek-v3.2`: aliases for DeepSeek V3.2; they
  start in reasoning mode when either `thinking=True` or
  `enable_thinking=True`.
- `deepseek-v4`: uses the same mode switches as the DeepSeek V3.2 parser.

The legacy names `qwen-qwq`, `intern-s1`, and `deepseek-r1` still map to
`default`, but emit a deprecation warning. GPT-OSS does not use a registered
reasoning parser; LMDeploy selects its specialized OpenAI Harmony response
parser automatically.

### Add a custom parser

Create a module under `lmdeploy/serve/parsers/reasoning_parser/` and register
the parser with `ReasoningParserManager`. A reasoning parser declares the
protocol's opening and closing tags and whether generation starts in reasoning
mode. The unified response parser handles streaming and complete-response
splitting.

```python
# lmdeploy/serve/parsers/reasoning_parser/example_reasoning_parser.py
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name="example")
class ExampleReasoningParser(ReasoningParser):
    """Parser for a model that emits <reasoning>...</reasoning>."""

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

Import the module from
`lmdeploy/serve/parsers/reasoning_parser/__init__.py` so registration runs
before CLI validation:

```python
from .example_reasoning_parser import ExampleReasoningParser  # noqa: F401
```

At server startup, LMDeploy validates that both declared tags are standalone
tokens in the model tokenizer. Replace the example tags with the model's exact
protocol tokens, or override `validate_tokenizer` when a model requires
different validation.

Then start the service with:

```bash
lmdeploy serve api_server "$MODEL_PATH" --reasoning-parser example
```
