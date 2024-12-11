# Serving LoRA

## Launch LoRA

LoRA is currently only supported by the PyTorch backend. Its deployment process is similar to that of other models, and you can view the commands using lmdeploy `serve api_server -h`. Among the parameters supported by the PyTorch backend, there are configuration options for LoRA.

```txt
PyTorch engine arguments:
  --adapters [ADAPTERS [ADAPTERS ...]]
                        Used to set path(s) of lora adapter(s). One can input key-value pairs in xxx=yyy format for multiple lora adapters. If only have one adapter, one can only input the path of the adapter.. Default:
                        None. Type: str
```

The user only needs to pass the Hugging Face model path of the LoRA weights in the form of a dictionary to `--adapters`.

```shell
lmdeploy serve api_server THUDM/chatglm2-6b --adapters mylora=chenchi/lora-chatglm2-6b-guodegang
```

After the service starts, you can find two available model names in the Swagger UI: ‘THUDM/chatglm2-6b’ and ‘mylora’. The latter is the key in the `--adapters` dictionary.

## Client usage

### CLI

When using the OpenAI endpoint, the `model` parameter can be used to select either the base model or a specific LoRA weight for inference. The following example chooses to use the provided `chenchi/lora-chatglm2-6b-guodegang` for inference.

```shell
curl -X 'POST' \
  'http://localhost:23334/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "mylora",
  "messages": [
    {
      "content": "hi",
      "role": "user"
    }
  ]
}'
```

And here is the output:

```json
{
  "id": "2",
  "object": "chat.completion",
  "created": 1721377275,
  "model": "mylora",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " 很高兴哪有什么赶凳儿？（按东北语说的“起早哇”），哦，东北人都学会外语了？",
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 17,
    "total_tokens": 43,
    "completion_tokens": 26
  }
}
```

### python

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = 'mylora'
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "user", "content": "hi"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

The printed response content is:

```txt
ChatCompletion(id='4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' 很高兴能够见到你哪，我也在辐射区开了个愣儿，你呢，还活着。', role='assistant', function_call=None, tool_calls=None))], created=1721377497, model='mylora', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=22, prompt_tokens=17, total_tokens=39))
```
