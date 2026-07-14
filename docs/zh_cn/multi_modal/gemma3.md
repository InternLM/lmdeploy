# Gemma3

## 简介

Gemma 是 Google 推出的轻量级、最先进的开放模型系列，采用与创建 Gemini 模型相同的研究和技术构建而成。Gemma3 模型是多模态模型，可处理文本和图像输入并生成文本输出，对预训练和指令微调均具有开源的权重。Gemma3 具有 128K 的大型上下文窗口，支持 140 多种语言，并且比以前的版本提供更多尺寸。Gemma3 模型非常适合各种文本生成和图像理解任务，包括问答、总结和推理。它们的尺寸相对较小，因此可以将其部署在资源有限的环境中，例如笔记本电脑、台式机或您自己的云基础设施，从而让每个人都能轻松访问最先进的 AI 模型，并帮助促进创新。

## 快速开始

请参考[安装文档](../get_started/installation.md)安装 LMDeploy。

### 准备

在使用 LMDeploy 部署 **Gemma3** 模型时，请安装最新的 transformers。

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)。

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image


if __name__ == "__main__":
    pipe = pipeline('google/gemma-3-12b-it')

    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```
