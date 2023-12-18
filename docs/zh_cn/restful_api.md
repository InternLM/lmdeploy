# Restful API

### 启动服务

用户将下面命令输出的 http url 复制到浏览器打开，详细查看所有的 API 及其使用方法。
请一定查看`http://{server_ip}:{server_port}`！！！
请一定查看`http://{server_ip}:{server_port}`！！！
请一定查看`http://{server_ip}:{server_port}`！！！
重要的事情说三遍。

```shell
lmdeploy serve api_server ./workspace 0.0.0.0 --server_port ${server_port} --instance_num 64 --tp 1
```

我们提供的 restful api，其中三个仿照 OpenAI 的形式。

- /v1/chat/completions
- /v1/models
- /v1/completions

不过，我们建议用户用我们提供的另一个 API: `/v1/chat/interactive`。
它有更好的性能，提供更多的参数让用户自定义修改。

### python

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

### Java/Golang/Rust

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

### cURL

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

### CLI client

restful api 服务可以通过客户端测试，例如

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client api_server_url
```

### webui

也可以直接用 webui 测试使用 restful-api。

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
# server_name 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: lmdeploy serve gradio http://localhost:23333 --server_name localhost --server_port 6006
lmdeploy serve gradio api_server_url --server_name ${gradio_ui_ip} --server_port ${gradio_ui_port}
```

### FAQ

1. 当返回结果结束原因为 `"finish_reason":"length"`，这表示回话长度超过最大值。如需调整会话支持的最大长度，可以通过启动`api_server`时，设置`--session_len`参数大小。

2. 当服务端显存 OOM 时，可以适当减小启动服务时的 `instance_num` 个数

3. 当同一个 `session_id` 的请求给 `/v1/chat/interactive` 函数后，出现返回空字符串和负值的 `tokens`，应该是 `session_id` 混乱了，可以先将交互模式关闭，再重新开启。

4. `/v1/chat/interactive` api 支持多轮对话, 但是默认关闭。`messages` 或者 `prompt` 参数既可以是一个简单字符串表示用户的单词提问，也可以是一段对话历史。
