# DeepSeek-VL2

## 简介

DeepSeek-VL2 是一系列先进的 MoE 视觉-语言模型，相较于其前身 DeepSeek-VL 有了显著的改进。
DeepSeek-VL2 在各种任务中展现出卓越的能力，包括但不限于视觉问答、OCR、文档/表格/图表理解以及视觉定位。

LMDeploy 目前在 Pytorch 引擎中支持 [deepseek-vl2-tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny), [deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) 和 [deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) 。

## 快速开始

请参考[安装文档](../get_started/installation.md)安装 LMDeploy。

### 准备

在使用 LMDeploy 部署 **DeepSeek-VL2** 模型时，您必须安装官方的 GitHub 仓库以及一些相关的第三方库。这是因为 LMDeploy 会复用官方仓库中提供的图像处理功能。

```
pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
pip install attrdict timm 'transformers<=4.48.0'
```

值得注意的是，如果使用 transformers>=4.48.0，可能会出现失败的情况，详情可以参考此 [Issue](https://github.com/deepseek-ai/DeepSeek-VL2/issues/45)。

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)。

为了构建有效的、包含图像输入的 DeepSeek-VL2 提示词，用户应手动插入 `<IMAGE_TOKEN>`

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image


if __name__ == "__main__":
    pipe = pipeline('deepseek-vl2-tiny')

    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('<IMAGE_TOKEN>describe this image', image))
    print(response)
```
