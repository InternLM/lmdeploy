# LLaVA

LMDeploy 支持以下 LLaVA 系列模型，具体如下表所示：

|                 模型                 | 大小 |   支持的推理引擎   |
| :----------------------------------: | :--: | :----------------: |
| llava-hf/Llava-interleave-qwen-7b-hf |  7B  | TurboMind, PyTorch |
|       llava-hf/llava-1.5-7b-hf       |  7B  | TurboMind, PyTorch |
|  llava-hf/llava-v1.6-mistral-7b-hf   |  7B  |      PyTorch       |
|   llava-hf/llava-v1.6-vicuna-7b-hf   |  7B  |      PyTorch       |
|   liuhaotian/llava-v1.6-vicuna-7b    |  7B  |     TurboMind      |
|   liuhaotian/llava-v1.6-mistral-7b   |  7B  |     TurboMind      |

接下来的章节将演示如何使用 LMDeploy 部署 LLaVA 模型，并以 [llava-hf/llava-interleave](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) 为例。

```{note}
自 0.6.4 之后，PyTorch 引擎移除了对 llava 原始模型的支持。我们建议使用它们对应的 transformers 格式的模型。这些模型可以在 https://huggingface.co/llava-hf 中找到
```

## 安装

请按照[安装指南](../get_started/installation.md)安装 LMDeploy。

或者，您也可以使用官方的 Docker 镜像：

```shell
docker pull openmmlab/lmdeploy:latest
```

## 离线推理

以下示例代码展示了 VLM pipeline 的基本用法。有关详细信息，请参考 [VLM 离线推理流程](./vl_pipeline.md)。

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

pipe = pipeline("llava-hf/llava-interleave-qwen-7b-hf", backend_config=TurbomindEngineConfig(cache_max_entry_count=0.5),
    gen_config=GenerationConfig(max_new_tokens=512))

image = load_image('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg')
prompt = 'Describe the image.'
print(f'prompt:{prompt}')
response = pipe((prompt, image))
print(response)
```

更多示例：

<details>
  <summary><b>多图片多轮对话，组合图片</b></summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('llava-hf/llava-interleave-qwen-7b-hf', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
```

</details>

## 在线服务

可以使用 `lmdeploy serve api_server` CLI 启动服务器：

```shell
lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
```

或者，使用前面提到的 Docker 镜像启动服务：

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
```

采用 Docker Compose 部署也是一种常见选择。在 lmdeploy 项目的根目录创建 `docker-compose.yml` 文件，如下：

```yaml
version: '3.5'

services:
  lmdeploy:
    container_name: lmdeploy
    image: openmmlab/lmdeploy:latest
    ports:
      - "23333:23333"
    environment:
      HUGGING_FACE_HUB_TOKEN: <secret>
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    stdin_open: true
    tty: true
    ipc: host
    command: lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
```

然后，可以执行以下命令启动服务：

```shell
docker-compose up -d
```

当运行 `docker logs -f lmdeploy` 后看到如下日志，说明服务启动成功：

```text
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on  http://0.0.0.0:23333  (Press CTRL+C to quit)
```

可以通过 `lmdeploy serve api_server -h` 查看 `lmdeploy serve api_server` 的参数详情。

关于 `api_server` 以及如何访问服务的更多信息可以在[这里](api_server_vl.md)找到。
