# Reasoning Outputs

对于支持推理能力的模型，比如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，LMDeploy 支持在服务中将推理的结果解析出来，并单独用
reasoning_content 记录推理内容。

## 使用示例

### DeepSeek R1

我们可以像启动其他模型的 api_server 服务一样启动 DeepSeek R1 的模型，只是不同的是，我们需要指定 `--reasoning-parser`。
在 `--reasoning-parser` 传参里，我们需要指定具体的 parser。

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

只需要在 `lmdeploy/serve/openai/reasoning_parser/reasoning_parser.py` 中添加一个类似的 parser 类即可。

```python
# import the required packages
from typing import Sequence, Union, Tuple, Optional

from lmdeploy.serve.openai.reasoning_parser import (
    ReasoningParser, ReasoningParserManager)
from lmdeploy.serve.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)

# define a reasoning parser and register it to lmdeploy
# the name list in register_module can be used
# in --reasoning-parser.
@ReasoningParserManager.register_module(["example"])
class ExampleParser(ReasoningParser):
    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Args:
            model_output (str): The model-generated string to extract reasoning content from.
            request (ChatCompletionRequest): he request object that was used to generate the model_output.

        Returns:
            reasoning_content (str | None): The reasoning content.
            final_output (str | None): The content.
        """
```

类似的，启动服务的命令就变成了：

```
lmdeploy serve api_server $model_path --reasoning-parser example
```
