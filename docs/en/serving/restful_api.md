# Restful API

## Launch Service

The user can open the http url print by the following command in a browser.

- **Please check the http url for the detailed api usage!!!**
- **Please check the http url for the detailed api usage!!!**
- **Please check the http url for the detailed api usage!!!**

```shell
lmdeploy serve api_server ./workspace --server-name 0.0.0.0 --server-port ${server_port} --tp 1
```

The parameters supported by api_server can be viewed through the command line `lmdeploy serve api_server -h`.

We provide some RESTful APIs. Three of them are in OpenAI format.

- /v1/chat/completions
- /v1/models
- /v1/completions

However, we recommend users try
our own api `/v1/chat/interactive` which provides more arguments for users to modify. The performance is comparatively better.

**Note** please, if you want to launch multiple requests, you'd better set different `session_id` for both
`/v1/chat/completions` and `/v1/chat/interactive` apis. Or, we will set them random values.

## Deploy http service with docker

LMDeploy offers [official docker image](https://hub.docker.com/r/openmmlab/lmdeploy/tags) for deployment. The image can be used to run OpenAI compatible server.

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server internlm/internlm2-chat-7b
```

Just like the previous section, user can try the Swagger UI with a web browser.

## python

We have integrated the client-side functionalities of these services into the `APIClient` class. Below are some examples demonstrating how to invoke the `api_server` service on the client side.

If you want to use the `/v1/chat/completions` endpoint, you can try the following code:

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
messages = [{"role": "user", "content": "Say this is a test!"}]
for item in api_client.chat_completions_v1(model=model_name, messages=messages):
    print(item)
```

For the `/v1/completions` endpoint. If you want to use the `/v1/completions` endpoint, you can try:

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)
```

Lmdeploy supports maintaining session histories on the server for `/v1/chat/interactive` api. We disable the
feature by default.

- On interactive mode, the chat history is kept on the server. In a multiple rounds of conversation, you should set
  `interactive_mode = True` and the same `session_id` (can't be -1, it's the default number) to `/v1/chat/interactive` for requests.
- On normal mode, no chat history is kept on the server.

The interactive mode can be controlled by the `interactive_mode` boolean parameter. The following is an example of normal mode. If you want to experience the interactive mode, simply pass in `interactive_mode=True`.

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
for item in api_client.chat_interactive_v1(prompt='hi'):
    print(item)
```

## Java/Golang/Rust

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

## cURL

cURL is a tool for observing the output of the api.

List Models:

```bash
curl http://{server_ip}:{server_port}/v1/models
```

Interactive Chat:

```bash
curl http://{server_ip}:{server_port}/v1/chat/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello! How are you?",
    "session_id": 1,
    "interactive_mode": true
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

## CLI client

There is a client script for restful api server.

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client api_server_url
```

## webui through gradio

You can also test restful-api through webui.

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: lmdeploy serve gradio http://localhost:23333 --server-name localhost --server-port 6006
lmdeploy serve gradio api_server_url --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
```

## webui through OpenAOE

You can use [OpenAOE](https://github.com/InternLM/OpenAOE) for seamless integration with LMDeploy.

```shell
pip install -U openaoe
openaoe -f /path/to/your/config-template.yaml
```

Please refer to the [guidance](https://github.com/InternLM/OpenAOE/blob/main/docs/tech-report/model_serving_by_lmdeploy/model_serving_by_lmdeploy.md) for more deploy information.

## FAQ

1. When user got `"finish_reason":"length"`, it means the session is too long to be continued. The session length can be
   modified by passing `--session_len` to api_server.

2. When OOM appeared at the server side, please reduce the `cache_max_entry_count` of `backend_config` when lanching the service.

3. When the request with the same `session_id` to `/v1/chat/interactive` got a empty return value and a negative `tokens`, please consider setting `interactive_mode=false` to restart the session.

4. The `/v1/chat/interactive` api disables engaging in multiple rounds of conversation by default. The input argument `prompt` consists of either single strings or entire chat histories.

5. If you need to adjust other default parameters of the session, such as the content of fields like system. You can directly pass in the initialization parameters of the [dialogue template](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py). For example, for the internlm-chat-7b model, you can set the `--meta-instruction` parameter when starting the `api_server`.

6. Regarding the stop words, we only support characters that encode into a single index. Furthermore, there may be multiple indexes that decode into results containing the stop word. In such cases, if the number of these indexes is too large, we will only use the index encoded by the tokenizer. If you want use a stop symbol that encodes into multiple indexes, you may consider performing string matching on the streaming client side. Once a successful match is found, you can then break out of the streaming loop.

## request distribution service

Please refer to our [request distributor server](./proxy_server.md)
