# Mllama

## 简介

[Llama3.2-VL](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) 是 Meta 发布的新视觉模型。

本文将以[meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)为例，演示使用 LMDeploy 部署 Mllama 系列多模态模型的方法

## 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy。

## 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('meta-llama/Llama-3.2-11B-Vision-Instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## 在线服务

### 服务启动

你可以通过 `lmdeploy serve api_server` CLI 工具启动服务：

```shell
lmdeploy serve api_server meta-llama/Llama-3.2-11B-Vision-Instruct
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
