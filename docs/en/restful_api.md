# Restful API

### Launch Service

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace 0.0.0.0 server_port --instance_num 32 --tp 1
```

Then, the user can open the swagger UI: `http://{server_ip}:{server_port}` for the detailed api usage.
We provide five restful api in total. Three of them are in OpenAI format. However, we recommend users try
our own api `generate` which provides more arguments for users to modify. The performance is comparatively better.

**Note** please, lmdeploy supports maintaining session histories on the server for `generate` api. We disable the
feature by default.

- On interactive mode, the chat history is kept on the server. In a multiple rounds of conversation, you should set
  the same `session_id` (can't be -1, it's the default number) to `generate` for the following requests.

  1. Set `sequence_start = True` and `sequence_end = False` for the first request,
  2. Set `sequence_start = False` and `sequence_end = False` for the later
     requests.
  3. Once the session length limit is reached, the output `finish_reason` would be `'length'`.
     Set `sequence_start = False` and `sequence_end = True` to end the session.
     Then restart the above steps for a new session.

- On normal mode, no chat history is kept on the server. Set
  `sequence_start = True` and `sequence_end = True` for all requests.

**Note** please, if you want to launch multiple requests, you'd better set different `session_id` for both
`/v1/chat/completions` and `/generate` apis. Or, we will set them random values.

### python

Here is an example for the above api.

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
messages = [{"role": "user", "content": "Say this is a test!"}]
for item in api_client.chat_completions_v1(model=model_name, messages = messages):
    print(item)

for item in api_client.generate(prompt='hi'):
    print(item)

for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)

for item in api_client.embeddings_v1(model=model_name, input='hi'):
    print(item)

outputs = api_client.completion_v1_concurrently(
        model_name, ['hi', 'Say this is a test!'])
print(outputs)
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
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

Text Completions:

```shell
curl http://{server_ip}:{server_port}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama",
  "prompt": "two steps to build a house:"
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
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
python -m lmdeploy.serve.openai.api_client api_server_url
```

### webui

You can also test restful-api through webui.

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006
python -m lmdeploy.serve.gradio.app api_server_url gradio_ui_ip gradio_ui_port
```

### FAQ

1. When user got `"finish_reason":"length"` which means the session is too long to be continued.
   Please add `"renew_session": true` into the next request.

2. When OOM appeared at the server side, please reduce the number of `instance_num` when lanching the service.

3. When the request with the same `session_id` to `generate` got a empty return value and a negative `tokens`, please consider setting `sequence_start=false` for the second question and the same for the afterwards.

4. Requests were previously being handled sequentially rather than concurrently. To resolve this issue,

   - kindly provide unique session_id values when calling the `generate` API or else your requests may be associated with client IP addresses

5. The `generate` api supports engaging in multiple rounds of conversation, where input argument `prompt` consists of either single strings or entire chat histories. However, if you want to turn the mode off and manage the chat history in clients, please set the input argument `sequence_end: true` when utilizing the `generate` function.
