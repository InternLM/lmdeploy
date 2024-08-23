# Phi-3 Vision

## Introduction

[Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) is a family of small language and multi-modal models from MicroSoft. LMDeploy supports the multi-modal models [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) and [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) in PyTorch engine.

## Quick Start

### Installation

Please install LMDeploy by following the [installation guide](../installation.md)

Then install the dependency [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

```shell
pip install flash-attn
```

### Offline inference pipeline

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('microsoft/Phi-3.5-vision-instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```
