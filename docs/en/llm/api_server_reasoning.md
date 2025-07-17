# Reasoning Outputs

For models that support reasoning capabilities, such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), LMDeploy supports parsing the reasoning results in the service and separately records the reasoning content using `reasoning_content`.

## Examples

### DeepSeek R1

We can start the DeepSeek R1 model's api_server service just like launching other models. The difference is that we need to specify --reasoning-parser\` parameter.

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

You only need to add a similar parser class in `lmdeploy/serve/openai/reasoning_parser/reasoning_parser.py`.

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

Similarly, the command to start the service becomes:

```
lmdeploy serve api_server $model_path --reasoning-parser example
```
