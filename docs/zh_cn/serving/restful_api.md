# Restful API

## 启动服务

用户将下面命令输出的 http url 复制到浏览器打开，详细查看所有的 API 及其使用方法。
请一定查看`http://{server_ip}:{server_port}`！！！
请一定查看`http://{server_ip}:{server_port}`！！！
请一定查看`http://{server_ip}:{server_port}`！！！
重要的事情说三遍。

```shell
lmdeploy serve api_server ./workspace --server-name 0.0.0.0 --server-port ${server_port} --tp 1
```

api_server 启动时支持的参数可以通过命令行`lmdeploy serve api_server -h`查看。

我们提供的 restful api，其中三个仿照 OpenAI 的形式。

- /v1/chat/completions
- /v1/models
- /v1/completions

不过，我们建议用户用我们提供的另一个 API: `/v1/chat/interactive`。
它有更好的性能，提供更多的参数让用户自定义修改。

## 用 docker 部署 http 服务

LMDeploy 提供了官方[镜像](https://hub.docker.com/r/openmmlab/lmdeploy/tags)。使用这个镜像，可以运行兼容 OpenAI 的服务。下面是使用示例：

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server internlm/internlm2-chat-7b
```

然后像上面一样使用浏览器试用 Swagger UI 即可。

## python

我们将这些服务的客户端功能集成在 `APIClient` 类中。下面是一些例子，展示如何在客户端调用 `api_server` 服务。
如果你想用 `/v1/chat/completions` 接口，你可以尝试下面代码：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
messages = [{"role": "user", "content": "Say this is a test!"}]
for item in api_client.chat_completions_v1(model=model_name, messages=messages):
    print(item)
```

如果你想用 `/v1/completions` 接口，你可以尝试：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)
```

LMDeploy 的 `/v1/chat/interactive` api 支持将对话内容管理在服务端，但是我们默认关闭。如果想尝试，请阅读以下介绍：

- 交互模式下，对话历史保存在 server。在一次完整的多轮对话中，所有请求设置`interactive_mode = True`, `session_id`保持相同 (不为 -1，这是缺省值)。
- 非交互模式下，server 不保存历史记录。

交互模式可以通过 `interactive_mode` 布尔量参数控制。下面是一个普通模式的例子，
如果要体验交互模式，将 `interactive_mode=True` 传入即可。

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
for item in api_client.chat_interactive_v1(prompt='hi'):
    print(item)
```

## Java/Golang/Rust

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

## cURL

cURL 也可以用于查看 API 的输出结果

查看模型列表：

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

restful api 服务可以通过客户端测试，例如

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client api_server_url
```

## webui through gradio

也可以直接用 webui 测试使用 restful-api。

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
# server_name 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: lmdeploy serve gradio http://localhost:23333 --server-name localhost --server-port 6006
lmdeploy serve gradio api_server_url --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
```

## webui through OpenAOE

可以使用 [OpenAOE](https://github.com/InternLM/OpenAOE) 无缝接入restful api服务.

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

5. 如需调整会话默认的其他参数，比如 system 等字段的内容，可以直接将[对话模板](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py)初始化参数传入。比如 internlm-chat-7b 模型，可以通过启动`api_server`时，设置`--meta-instruction`参数。

6. 关于停止符，我们只支持编码后为单个 index 的字符。此外，可能存在多种 index 都会解码出带有停止符的结果。对于这种情况，如果这些 index 数量太多，我们只会采用 tokenizer 编码出的 index。而如果你想要编码后为多个 index 的停止符，可以考虑在流式客户端做字符串匹配，匹配成功后跳出流式循环即可。

## 多机并行服务

请参考我们的 [请求分发服务器](./proxy_server.md)

## 自定义对话模板

自定义对话模板，请参考[chat_template.md](../advance/chat_template.md)
