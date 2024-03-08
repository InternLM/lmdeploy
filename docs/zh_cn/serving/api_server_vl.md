# 部署 VL 的类 openai 服务

本文主要介绍单个VL模型在单机多卡环境下，部署兼容 openai 接口服务的方式，以及服务接口的用法。为行文方便，我们把该服务名称为 `api_server`。对于多模型的并行服务，请阅读[请求分发服务器](./proxy_server.md)一文。

在这篇文章中， 我们首先介绍服务启动的两种方法，你可以根据应用场景，选择合适的。

其次，我们重点介绍服务的 RESTful API 定义，以及接口使用的方式，并展示如何通过 Swagger UI、LMDeploy CLI 工具体验服务功能

最后，向大家演示把服务接入到 WebUI 的方式，你可以参考它简单搭建一个演示 demo。

## 启动服务

以 huggingface hub 上的 [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) 模型为例，你可以任选以下方式之一，启动推理服务。

### 方式一：使用 lmdeploy cli 工具

```shell
lmdeploy serve api_server liuhaotian/llava-v1.5-7b --server-port 23333 --task vision-language
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
    lmdeploy serve api_server liuhaotian/llava-v1.5-7b --task vision-language
```

在这个例子中，`lmdeploy server api_server` 的命令参数与方式一一致。

## RESTful API

LMDeploy 的 RESTful API 兼容了 OpenAI 以下 3 个接口：

- /v1/chat/completions
- /v1/models
- /v1/completions

其中使用图片交互的接口是 `/v1/chat/completions`，与 OpenAI 的一致。
服务启动后，你可以在浏览器中打开网页 http://0.0.0.0:23333，通过 Swagger UI 查看接口的详细说明，并且也可以直接在网页上操作，体验每个接口的用法，如下图所示。

![swagger_ui](https://github.com/InternLM/lmdeploy/assets/4560679/b891dd90-3ffa-4333-92b2-fb29dffa1459)

若需要把服务集成到自己的项目或者产品中，我们推荐以下用法：

### 使用 openai 接口

以下代码是通过 openai 包使用 `v1/chat/completions` 服务的例子。运行之前，请先安装 openai 包: `pip install openai`。

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')

response = client.chat.completions.create(
    model='llama-2',
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

### 使用 lmdeploy `APIClient` 接口

如果你想用 `/v1/chat/completions` 接口，你可以尝试下面代码：

```python
from lmdeploy.serve.openai.api_client import APIClient

api_client = APIClient(f'http://0.0.0.0:23333')
model_name = api_client.available_models[0]
messages = [{
    'role':
    'user',
    'content': [{
        'type': 'text',
        'text': 'Describe the image please',
    }, {
        'type': 'image_url',
        'image_url': {
            'url':
            'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing.jpeg',
        },
    }]
}]
for item in api_client.chat_completions_v1(model=model_name,
                                           messages=messages):
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
