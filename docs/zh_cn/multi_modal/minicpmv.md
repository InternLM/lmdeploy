# MiniCPM-V

## 简介

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出。 LMDeploy 支持了 [openbmb/MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) 模型，通过 TurboMind 引擎推理。

## 快速开始

### 安装

使用 pip(Python 3.8+) 安装 LMDeploy，更多安装方式参考 [安装](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started.html#id2)。

```shell
pip install lmdeploy
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](https://lmdeploy.readthedocs.io/zh-cn/latest/inference/vl_pipeline.html#vlm-pipeline)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('openbmb/MiniCPM-Llama3-V-2_5')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```
