# Restful API

### Launch Service

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace server_name server_port --instance_num 32 --tp 1
```

Then, the user can open the swagger UI: http://{server_name}:{server_port}/docs for the detailed api usage.
We provide four restful api in total. Three of them are in OpenAI format. However, we recommend users try
our own api which provides more arguments for users to modify. The performance is comparatively better.

### python

Here is an example for our own api `generate`.

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

### Golang/Rust

Golang can also build a http request to use the service. You may refer
to [the blog](https://pkg.go.dev/net/http) for details to build own client.
Besides, Rust supports building a client in [many ways](https://blog.logrocket.com/best-rust-http-client/).

### cURL

cURL is a tool for observing the output of the api.

List Models:

```bash
curl http://{server_name}:{server_port}/v1/models
```

Generate:

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

### FAQ

1. When user got `"finish_reason":"length"` which means the session is too long to be continued.
   Please add `"renew_session": true` into the next request.

2. When OOM appeared at the server side, please reduce the number of `instance_num` when lanching the service.

3. When the request with the same `instace_id` to `generate` got a empty return value and a negative `tokens`, please consider setting `sequence_start=false` for the second question and the same for the afterwards.
