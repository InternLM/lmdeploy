# LoRA 推理服务

## 启动 LoRA 服务

LoRA 目前只有 pytorch 后端支持。它的服务化，和其他模型服务化一样，命令都可以用 `lmdeploy serve api_server -h` 查看。其中 pytorch 后端支持的参数就有 LoRA 的配置内容。

```txt
PyTorch engine arguments:
  --adapters [ADAPTERS [ADAPTERS ...]]
                        Used to set path(s) of lora adapter(s). One can input key-value pairs in xxx=yyy format for multiple lora adapters. If only have one adapter, one can only input the path of the adapter.. Default:
                        None. Type: str
```

用户只需要将 lora 权重的 huggingface 模型路径通过字典的形式传入 `--adapters` 即可。

```shell
lmdeploy serve api_server THUDM/chatglm2-6b --adapters mylora=chenchi/lora-chatglm2-6b-guodegang
```

服务启动后，可以在 Swagger UI 中查询到两个可用的模型名字：“THUDM/chatglm2-6b” 和 “mylora”。后者是 `--adapters` 字典的 key。

## 客户端使用

### CLI

使用时，OpenAI 接口参数 `model` 可以用来选择使用基础模型还是某个 lora 权重用于推理。下面的例子就选择使用了传入的 `chenchi/lora-chatglm2-6b-guodegang` 用于推理。

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

可以得到一个这个 lora 权重特有的回复：

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

打印的响应内容为：

```txt
ChatCompletion(id='4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' 很高兴能够见到你哪，我也在辐射区开了个愣儿，你呢，还活着。', role='assistant', function_call=None, tool_calls=None))], created=1721377497, model='mylora', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=22, prompt_tokens=17, total_tokens=39))
```
