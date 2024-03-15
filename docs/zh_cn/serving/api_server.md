# 部署 LLMs 类 openai 服务

本文主要介绍单个模型在单机多卡环境下，部署兼容 openai 接口服务的方式，以及服务接口的用法。为行文方便，我们把该服务名称为 `api_server`。对于多模型的并行服务，请阅读[请求分发服务器](./proxy_server.md)一文。

在这篇文章中， 我们首先介绍服务启动的两种方法，你可以根据应用场景，选择合适的。

其次，我们重点介绍服务的 RESTful API 定义，以及接口使用的方式，并展示如何通过 Swagger UI、LMDeploy CLI 工具体验服务功能

最后，向大家演示把服务接入到 WebUI 的方式，你可以参考它简单搭建一个演示 demo。

## 启动服务

以 huggingface hub 上的 [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) 模型为例，你可以任选以下方式之一，启动推理服务。

### 方式一：使用 lmdeploy cli 工具

```shell
lmdeploy serve api_server internlm/internlm2-chat-7b --server-port 23333
```

api_server 启动时的参数可以通过命令行`lmdeploy serve api_server -h`查看。
比如，`--tp` 设置张量并行，`--session-len` 设置推理的最大上下文窗口长度，`--cache-max-entry-count` 调整 k/v cache 的内存使用比例等等。

### 方式二：使用 docker

使用 LMDeploy 官方[镜像](https://hub.docker.com/r/openmmlab/lmdeploy/tags)，可以运行兼容 OpenAI 的服务。下面是使用示例：

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server internlm/internlm2-chat-7b
```

在这个例子中，`lmdeploy server api_server` 的命令参数与方式一一致。

## RESTful API

LMDeploy 的 RESTful API 兼容了 OpenAI 以下 3 个接口：

- /v1/chat/completions
- /v1/models
- /v1/completions

此外，LMDeploy 还定义了 `/v1/chat/interactive`，用来支持交互式推理。交互式推理的特点是不用像`v1/chat/completions`传入用户对话历史，因为对话历史会被缓存在服务端。
这种方式在多轮次的长序列推理时，拥有很好的性能。

服务启动后，你可以在浏览器中打开网页 http://0.0.0.0:23333，通过 Swagger UI 查看接口的详细说明，并且也可以直接在网页上操作，体验每个接口的用法，如下图所示。

![swagger_ui](https://github.com/InternLM/lmdeploy/assets/4560679/b891dd90-3ffa-4333-92b2-fb29dffa1459)

也可以使用 LMDeploy 自带的 CLI 工具，在控制台验证服务的正确性。

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client ${api_server_url}
```

若需要把服务集成到自己的项目或者产品中，我们推荐以下用法：

### 使用 openai 接口

以下代码是通过 openai 包使用 `v1/chat/completions` 服务的例子。运行之前，请先安装 openai 包: `pip install openai`。

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": " provide three suggestions about time management"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

关于其他 openai 接口的调用，也可以如法炮制。详情请参考 openai 官方[文档](https://platform.openai.com/docs/guides/text-generation)

### 使用 lmdeploy `APIClient` 接口

如果你想用 `/v1/chat/completions` 接口，你可以尝试下面代码：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient(f'http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
messages = [{"role": "user", "content": "Say this is a test!"}]
for item in api_client.chat_completions_v1(model=model_name, messages=messages):
    print(item)
```

如果你想用 `/v1/completions` 接口，你可以尝试：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient(f'http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)
```

关于 `/v1/chat/interactive` 接口，我们默认是关闭的。在使用时，请设置`interactive_mode = True`打开它。否则，它会退化为 openai 接口。

在交互式推理中，每个对话序列的 id 必须唯一，所有属于该独立的对话请求，必须使用相同的 id。这里的 id 对应与接口中的 `session_id`。
比如，一个对话序列中，有 10 轮对话请求，那么每轮对话请求中的 `session_id` 都要相同。

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

### 使用 Java/Golang/Rust

可以使用代码生成工具 [openapi-generator-cli](https://github.com/OpenAPITools/openapi-generator-cli) 将 `http://{server_ip}:{server_port}/openapi.json` 转成 java/rust/golang 客户端。
下面是一个使用示例：

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

### 使用 cURL

cURL 也可以用于查看 API 的输出结果

- 查看模型列表 `v1/models`

```bash
curl http://{server_ip}:{server_port}/v1/models
```

- 对话 `v1/chat/completions`

```bash
curl http://{server_ip}:{server_port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-chat-7b",
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

- 文本补全 `v1/completions`

```shell
curl http://{server_ip}:{server_port}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama",
  "prompt": "two steps to build a house:"
}'
```

- 交互式对话 `v1/chat/interactive`

```bash
curl http://{server_ip}:{server_port}/v1/chat/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello! How are you?",
    "session_id": 1,
    "interactive_mode": true
  }'
```

## 接入 WebUI

LMDeploy 提供 gradio 和 [OpenAOE](https://github.com/InternLM/OpenAOE) 两种方式，为 api_server 接入 WebUI。

### 方式一：通过 gradio 接入

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
# server_name 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: lmdeploy serve gradio http://localhost:23333 --server-name localhost --server-port 6006
lmdeploy serve gradio api_server_url --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
```

### 方式二：通过 OpenAOE 接入

```shell
pip install -U openaoe
openaoe -f /path/to/your/config-template.yaml
```

具体信息请参考 [部署说明](https://github.com/InternLM/OpenAOE/blob/main/docs/tech-report/model_serving_by_lmdeploy/model_serving_by_lmdeploy.md).

## FAQ

1. 当返回结果结束原因为 `"finish_reason":"length"`，这表示回话长度超过最大值。如需调整会话支持的最大长度，可以通过启动`api_server`时，设置`--session_len`参数大小。

2. 当服务端显存 OOM 时，可以适当减小启动服务时的 `backend_config` 的 `cache_max_entry_count` 大小

3. 当同一个 `session_id` 的请求给 `/v1/chat/interactive` 函数后，出现返回空字符串和负值的 `tokens`，应该是 `session_id` 混乱了，可以先将交互模式关闭，再重新开启。

4. `/v1/chat/interactive` api 支持多轮对话, 但是默认关闭。`messages` 或者 `prompt` 参数既可以是一个简单字符串表示用户的单词提问，也可以是一段对话历史。

5. 关于停止符，我们只支持编码后为单个 index 的字符。此外，可能存在多种 index 都会解码出带有停止符的结果。对于这种情况，如果这些 index 数量太多，我们只会采用 tokenizer 编码出的 index。而如果你想要编码后为多个 index 的停止符，可以考虑在流式客户端做字符串匹配，匹配成功后跳出流式循环即可。

6. 自定义对话模板，请参考[chat_template.md](../advance/chat_template.md)
