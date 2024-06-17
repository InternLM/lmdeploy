# MiniCPM-V

## Introduction

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) is a series of end-side multimodal LLMs (MLLMs) designed for vision-language understanding. LMDeploy supports MiniCPM-Llama3-V-2_5 model [openbmb/MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) in TurboMind engine.

## Quick Start

Install LMDeploy with pip (Python 3.8+). Refer to [Installation](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation) for more.

```shell
pip install lmdeploy
```

### Offline inference pipeline

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#vlm-offline-inference-pipeline)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('openbmb/MiniCPM-Llama3-V-2_5')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```
