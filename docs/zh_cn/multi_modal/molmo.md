# Qwen2-VL

LMDeploy 支持 Molmo 系列模型，具体如下：

|      Model      | Size | Supported Inference Engine |
| :-------------: | :--: | :------------------------: |
| Molmo-7B-D-0924 |  7B  |         TurboMind          |
|  Molmo-72-0924  | 72B  |         TurboMind          |

本文将以[Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) 为例，演示使用 LMDeploy 部署 Molmo 系列模型的方法

## 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy。

## 离线推理

以下是使用 pipeline 进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('allenai/Molmo-7B-D-0924')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

更多例子如下：

<details>
  <summary>
    <b>多图多轮对话</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('Qwen/Qwen2-VL-2B-Instruct', log_level='INFO')
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

你可以通过 `lmdeploy serve api_server` CLI 工具启动服务：

```shell
lmdeploy serve api_server Qwen/Qwen2-VL-2B-Instruct
```

也可以基于 docker image 启动服务：

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:qwen2vl \
    lmdeploy serve api_server Qwen/Qwen2-VL-2B-Instruct
```

如果日志中有如下信息，就表明服务启动成功了。

```text
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on  http://0.0.0.0:23333  (Press CTRL+C to quit)
```

有关 `lmdeploy serve api_server` 的详细参数可以通过`lmdeploy serve api_server -h`查阅。

关于 `api_server` 更多的介绍，以及访问 `api_server` 的方法，请阅读[此处](api_server_vl.md)
