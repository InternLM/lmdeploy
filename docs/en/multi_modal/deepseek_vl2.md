# DeepSeek-VL2

## Introduction

DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL.
DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding.

LMDeploy supports [deepseek-vl2-tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny), [deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) and [deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) in PyTorch engine.

## Quick Start

Install LMDeploy by following the [installation guide](../get_started/installation.md).

### Prepare

When deploying the **DeepSeek-VL2** model using LMDeploy, you must download the official GitHub repository. This is because LMDeploy reuses the image processing functions provided in the official repository.

```
pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
pip install attrdict
```

Worth noticing that it may fail with `transformers>=4.48.0`, as known in this [issue](https://github.com/deepseek-ai/DeepSeek-VL2/issues/45).

### Offline inference pipeline

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md).

To construct valid DeepSeek-VL2 prompts with image inputs, users should insert `<IMAGE_TOKENS>` manually.

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image


if __name__ == "__main__":
    pipe = pipeline('deepseek-vl2-tiny')

    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('<IMAGE_TOKEN>describe this image', image))
    print(response)
```
