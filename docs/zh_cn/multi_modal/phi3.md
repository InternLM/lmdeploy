# Phi-3 Vision

## 简介

[Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) 是微软发布的轻量级系列模型。LMDeploy 在 PyTorch 引擎侧支持了其中的多模态模型 [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) 和 [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) 。

## 快速开始

### 安装

请参考[安装文档](../installation.md)安装 LMDeploy

此外，还需安装其依赖 [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

```shell
pip install flash-attn
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('microsoft/Phi-3.5-vision-instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```
