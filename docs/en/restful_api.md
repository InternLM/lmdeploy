# Restful API

### Launch Service

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace 0.0.0.0 server_port --instance_num 32 --tp 1
```

Then, the user can open the swagger UI: `http://{server_ip}:{server_port}` for the detailed api usage.
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
                           session_id: int,
                           request_output_len: int,
                           stream: bool = True,
                           sequence_start: bool = True,
                           sequence_end: bool = True,
                           ignore_eos: bool = False) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    pload = {
        'prompt': prompt,
        'stream': stream,
        'session_id': session_id,
        'request_output_len': request_output_len,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'ignore_eos': ignore_eos
    }
    response = requests.post(
        api_url, headers=headers, json=pload, stream=stream)
    for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            output = data['text']
            tokens = data['tokens']
            yield output, tokens


for output, tokens in get_streaming_response(
        "Hi, how are you?", "http://{server_ip}:{server_port}/generate", 0,
        512):
    print(output, end='')
```

### Java/Golang/Rust

May use [openapi-generator-cli](https://github.com/OpenAPITools/openapi-generator-cli) to convert `http://{server_ip}:{server_port}/openapi.json` to java/rust/golang client.
Here is an example:

```shell
$ docker run -it --rm -v ${PWD}:/local openapitools/openapi-generator-cli generate -i /local/openapi.json -g rust -o /local/rust

$ ls rust/*
rust/Cargo.toml  rust/git_push.sh  rust/README.md

rust/docs:
ChatCompletionRequest.md  EmbeddingsRequest.md  HttpValidationError.md  LocationInner.md  Prompt.md
DefaultApi.md             GenerateRequest.md    Input.md                Messages.md       ValidationError.md

rust/src:
apis  lib.rs  models
```

### cURL

cURL is a tool for observing the output of the api.

List Models:

```bash
curl http://{server_ip}:{server_port}/v1/models
```

Generate:

```bash
curl http://{server_ip}:{server_port}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello! How are you?",
    "session_id": 1,
    "sequence_start": true,
    "sequence_end": true
  }'
```

Chat Completions:

```bash
curl http://{server_ip}:{server_port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "messages": [{"role": "user", "content": "Hello! Ho are you?"}]
  }'
```

Embeddings:

```bash
curl http://{server_ip}:{server_port}/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "input": "Hello world!"
  }'
```

### CLI client

There is a client script for restful api server.

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
python -m lmdeploy.serve.openai.api_client restful_api_url
```

### webui

You can also test restful-api through webui.

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006 --restful_api True
python -m lmdeploy.serve.gradio.app restful_api_url server_ip --restful_api True
```

### FAQ

1. When user got `"finish_reason":"length"` which means the session is too long to be continued.
   Please add `"renew_session": true` into the next request.

2. When OOM appeared at the server side, please reduce the number of `instance_num` when lanching the service.

3. When the request with the same `session_id` to `generate` got a empty return value and a negative `tokens`, please consider setting `sequence_start=false` for the second question and the same for the afterwards.

4. Requests were previously being handled sequentially rather than concurrently. To resolve this issue,

   - kindly provide unique session_id values when calling the `generate` API or else your requests may be associated with client IP addresses

5. Both `generate` api and `v1/chat/completions` upport engaging in multiple rounds of conversation, where input `prompt` or `messages` consists of either single strings or entire chat histories.These inputs are interpreted using multi-turn dialogue modes. However, ff you want to turn the mode of and manage the chat history in clients, please the parameter `sequence_end: true` when utilizing the `generate` function, or specify `renew_session: true` when making use of `v1/chat/completions`
