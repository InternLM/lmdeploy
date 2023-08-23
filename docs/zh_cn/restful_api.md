# Restful API

### 启动服务

运行脚本

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace server_name server_port --instance_num 32 --tp 1
```

然后用户可以打开 swagger UI: `http://{server_name}:{server_port}` 详细查看所有的 API 及其使用方法。
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

### Golang/Rust

Golang 也可以建立 http 请求使用启动的服务，用户可以参考[这篇博客](https://pkg.go.dev/net/http)构建自己的客户端。
Rust 也有许多[方法](https://blog.logrocket.com/best-rust-http-client/)构建客户端，使用服务。

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

### CLI client

restful api 服务可以通过客户端测试，例如

```shell
# restful_api_url 就是 api_server 产生的，比如 http://localhost:23333
python -m lmdeploy.serve.openai.api_client restful_api_url
```

### webui

也可以直接用 webui 测试使用 restful-api。

```shell
# restful_api_url 就是 api_server 产生的，比如 http://localhost:23333
# server_name 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006 --restful_api True
python -m lmdeploy.serve.gradio.app restful_api_url server_name --restful_api True
```

### FAQ

1. 当返回结果结束原因为 `"finish_reason":"length"`，这表示回话长度超过最大值。
   请添加 `"renew_session": true` 到下一次请求中。

2. 当服务端显存 OOM 时，可以适当减小启动服务时的 `instance_num` 个数

3. 当同一个 `instance_id` 的请求给 `generate` 函数后，出现返回空字符串和负值的 `tokens`，应该是第二次问话没有设置 `sequence_start=false`
