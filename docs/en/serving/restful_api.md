# Serving with OpenAI Compatible Server

This article primarily discusses the deployment of a single LLM model across multiple GPUs on a single node, providing a service that is compatible with the OpenAI interface, as well as the usage of the service API.
For the sake of convenience, we refer to this service as `api_server`. Regarding parallel services with multiple models, please refer to the guide about [Request Distribution Server](./proxy_server.md).

In the following sections, we will first introduce two methods for starting the service, choosing the appropriate one based on your application scenario.

Next, we focus on the definition of the service's RESTful API, explore the various ways to interact with the interface, and demonstrate how to try the service through the Swagger UI or LMDeploy CLI tools.

Finally, we showcase how to integrate the service into a WebUI, providing you with a reference to easily set up a demonstration demo.

## Launch Service

Take the [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) model hosted on huggingface hub as an example, you can choose one the following methods to start the service.

### Option 1: Launching with lmdeploy CLI

```shell
lmdeploy serve api_server internlm/internlm2-chat-7b --server-port 23333
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.

### Option 2: Deploying with docker

With LMDeploy [official docker image](https://hub.docker.com/r/openmmlab/lmdeploy/tags), you can run OpenAI compatible server as follows:

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server internlm/internlm2-chat-7b
```

The parameters of `api_server` are the same with that mentioned in "[option 1](#option-1-launching-with-lmdeploy-cli)" section

## RESTful API

LMDeploy's RESTful API is compatible with the following three OpenAI interfaces:

- /v1/chat/completions
- /v1/models
- /v1/completions

Additionally, LMDeploy also defines `/v1/chat/interactive` to support interactive inference. The feature of interactive inference is that there's no need to pass the user conversation history as required by `v1/chat/completions`, since the conversation history will be cached on the server side. This method boasts excellent performance during multi-turn long context inference.

You can overview and try out the offered RESTful APIs by the website `http://0.0.0.0:23333` as shown in the below image after launching the service successfully.

![swagger_ui](https://github.com/InternLM/lmdeploy/assets/4560679/b891dd90-3ffa-4333-92b2-fb29dffa1459)

Or, you can use the LMDeploy's built-in CLI tool to verify the service correctness right from the console.

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client ${api_server_url}
```

If you need to integrate the service into your own projects or products, we recommend the following approach:

### Integrate with `OpenAI`

Here is an example of interaction with the endpoint `v1/chat/completions` service via the openai package.
Before running it, please install the openai package by `pip install openai`

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)

response = client.chat.completions.create(
  model="internlm2-chat-7b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": " provide three suggestions about time management"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

You can invoke other OpenAI interfaces using similar methods. For more detailed information, please refer to the [OpenAI API guide](https://platform.openai.com/docs/guides/text-generation)

### Integrate with lmdeploy `APIClient`

Below are some examples demonstrating how to visit the service through `APIClient`

If you want to use the `/v1/chat/completions` endpoint, you can try the following code:

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
messages = [{"role": "user", "content": "Say this is a test!"}]
for item in api_client.chat_completions_v1(model=model_name, messages=messages):
    print(item)
```

For the `/v1/completions` endpoint, you can try:

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)
```

As for `/v1/chat/interactive`，we disable the feature by default. Please open it by setting `interactive_mode = True`. If you don't, it falls back to openai compatible interfaces.

Keep in mind that `session_id` indicates an identical sequence and all requests belonging to the same sequence must share the same `session_id`.
For instance, in a sequence with 10 rounds of chatting requests, the `session_id` in each request should be the same.

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient(f'http://{server_ip}:{server_port}')
messages = [
    "hi, what's your name?",
    "who developed you?",
    "Tell me more about your developers",
    "Summarize the information we've talked so far"
]
for message in messages:
    for item in api_client.chat_interactive_v1(prompt=message,
                                               session_id=1,
                                               interactive_mode=True,
                                               stream=False):
        print(item)
```

### Integrate with Java/Golang/Rust

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

### Integrate with cURL

cURL is a tool for observing the output of the RESTful APIs.

- list served models `v1/models`

```bash
curl http://{server_ip}:{server_port}/v1/models
```

- chat `v1/chat/completions`

```bash
curl http://{server_ip}:{server_port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

- text completions `v1/completions`

```shell
curl http://{server_ip}:{server_port}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama",
  "prompt": "two steps to build a house:"
}'
```

- interactive chat `v1/chat/interactive`

```bash
curl http://{server_ip}:{server_port}/v1/chat/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello! How are you?",
    "session_id": 1,
    "interactive_mode": true
  }'
```

## Integrate with WebUI

LMDeploy utilizes `gradio` or [OpenAOE](https://github.com/InternLM/OpenAOE) to integrate a web ui for `api_server`

### Option 1: gradio

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: lmdeploy serve gradio http://localhost:23333 --server-name localhost --server-port 6006
lmdeploy serve gradio api_server_url --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
```

### Option 2: OpenAOE

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
