# Restful API

### 启动服务

运行脚本

```shell
python lmdeploy/serve/openai/api_server.py ./workspace server_name server_port
```

然后用户可以打开 http://{server_name}:{server_port}/docs 详细查看所有的 API 及其使用方法。
我们一共提供四个 restful api，其中三个仿照 OpenAI 的形式。不过，我们建议用户用我们提供的另一个 API: `generate`。
它有更好的性能，提供更多的参数让用户自定义修改。

### python

这是一个 python 示例，展示如何使用 `generate`。

```python
import json
import requests
from typing import Iterable, List


def get_streaming_response(prompt: str,
                           api_url: str,
                           instance_id: int,
                           request_output_len: int,
                           stream: bool = True,
                           sequence_start: bool = True,
                           sequence_end: bool = True,
                           ignore_eos: bool = False) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    pload = {
        'prompt': prompt,
        'stream': stream,
        'instance_id': instance_id,
        'request_output_len': request_output_len,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'ignore_eos': ignore_eos
    }
    response = requests.post(
        api_url, headers=headers, json=pload, stream=stream)
    for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b'\0'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            output = data['text']
            tokens = data['tokens']
            yield output, tokens


for output, tokens in get_streaming_response(
        "Hi, how are you?", "http://{server_name}:{server_port}/generate", 0,
        512):
    print(output, end='')
```

### cURL

cURL 也可以用于查看 API 的输出结果

查看模型列表：

```bash
curl http://{server_name}:{server_port}/v1/models
```

使用 generate:

```bash
curl http://{server_name}:{server_port}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "prompt": "Hello! Ho are you?",
    "sequence_start": true,
    "sequence_end": true
  }'
```

Chat Completions:

```bash
curl http://{server_name}:{server_port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "messages": [{"role": "user", "content": "Hello! Ho are you?"}]
  }'
```

Embeddings:

```bash
curl http://{server_name}:{server_port}/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "input": "Hello world!"
  }'
```
