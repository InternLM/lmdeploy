# Phi-3 Vision

## 简介

[Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) 是微软发布的轻量级系列模型，LMDeploy支持了其中的多模态模型如下：

|                                                Model                                                | Size | Supported Inference Engine |
| :-------------------------------------------------------------------------------------------------: | :--: | :------------------------: |
| [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | 4.2B |          PyTorch           |
|    [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)    | 4.2B |          PyTorch           |

本文将以[microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)为例，演示使用 LMDeploy 部署 Phi-3 系列多模态模型的方法

## 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy，并安装该模型的依赖。

```shell
# 建议从https://github.com/Dao-AILab/flash-attention/releases寻找和环境匹配的whl包
pip install flash-attn
```

## 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('microsoft/Phi-3.5-vision-instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## 在线服务

### 服务启动

你可以通过 `lmdeploy serve api_server` CLI 工具启动服务：

```shell
lmdeploy serve api_server microsoft/Phi-3.5-vision-instruct
```

### 使用 openai 接口

以下代码是通过 openai 包使用 `v1/chat/completions` 服务的例子。运行之前，请先安装 openai 包: `pip install openai`。

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
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
                'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```
