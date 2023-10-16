# Restful API

### 启动服务

运行脚本

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace 0.0.0.0 server_port --instance_num 32 --tp 1
```

然后用户可以打开 swagger UI: `http://{server_ip}:{server_port}` 详细查看所有的 API 及其使用方法。
我们一共提供五个 restful api，其中四个仿照 OpenAI 的形式。不过，我们建议用户用我们提供的另一个 API: `generate`。
它有更好的性能，提供更多的参数让用户自定义修改。

**注意**，LMDeploy 的 `generate` api 支持将对话内容管理在服务端，但是我们默认关闭。如果想尝试，请阅读以下介绍：

- 交互模式下，对话历史保存在 server。在一次完整的多轮对话中，所有请求设置一个相同 `session_id`(不为 -1，这是缺省值)。
- 非交互模式下，server 不保存历史记录。

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

对于 `/v1/completions` 接口，我们还特别提供了在单客户端批量推理接口，可以接受传入一个 `prompt` 列表。如果你想用 `/v1/completions` 接口，你可以尝试：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.completions_v1(model=model_name, prompt='hi'):
    print(item)
outputs = api_client.completion_v1_concurrently(
        model_name, ['hi', 'Say this is a test!'])
print(outputs)
```

类似地，如果要使用 `/v1/embeddings` 接口，也可以用 `APIClient`：

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
model_name = api_client.available_models[0]
for item in api_client.embeddings_v1(model=model_name, input='hi'):
    print(item)
```

对于`generate` 接口，我们提供两个模式，一个是普通模式，默认开启，一个是交互对话模式，默认关闭。模式可以通过 `interactive_mode` 布尔量参数控制。下面是一个普通模式的例子，
如果要体验交互模式，将 `interactive_mode=True` 传入即可。

```python
from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://{server_ip}:{server_port}')
for item in api_client.generate(prompt='hi'):
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

使用 generate:

```bash
curl http://{server_ip}:{server_port}/generate \
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

restful api 服务可以通过客户端测试，例如

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
python -m lmdeploy.serve.openai.api_client api_server_url
```

### webui

也可以直接用 webui 测试使用 restful-api。

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
# server_ip 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006
python -m lmdeploy.serve.gradio.app api_server_url gradio_ui_ip gradio_ui_port
```

### FAQ

1. 当返回结果结束原因为 `"finish_reason":"length"`，这表示回话长度超过最大值。
   请添加 `"renew_session": true` 到下一次请求中。

2. 当服务端显存 OOM 时，可以适当减小启动服务时的 `instance_num` 个数

3. 当同一个 `session_id` 的请求给 `generate` 函数后，出现返回空字符串和负值的 `tokens`，应该是 `session_id` 混乱了，可以先将交互模式关闭，再重新开启。

4. `generate` api 支持多轮对话, 但是默认关闭。`messages` 或者 `prompt` 参数既可以是一个简单字符串表示用户的单词提问，也可以是一段对话历史。
