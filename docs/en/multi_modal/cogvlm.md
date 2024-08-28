# CogVLM

## Introduction

CogVLM is a powerful open-source visual language model (VLM). LMDeploy supports CogVLM-17B models like [THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) and CogVLM2-19B models like [THUDM/cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) in PyTorch engine.

## Quick Start

### Install

Install torch, torchvision and xformers for CogVLM by referring to [Pytorch](https://pytorch.org/get-started) and [installing-xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers)

```shell
# cuda 11.8
pip install torch==2.2.2 torchvision==0.17.2 xformers==0.0.26 --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1
pip install torch==2.2.2 torchvision==0.17.2 xformers==0.0.26 --index-url https://download.pytorch.org/whl/cu121
```

Install LMDeploy by following the [installation guide](../get_started/installation.md)

### Prepare

When deploying the **CogVLM** model using LMDeploy, it is necessary to download the model first, as the **CogVLM** model repository does not include the tokenizer model.
However, this step is not required for **CogVLM2**.

Taking one **CogVLM** model `cogvlm-chat-hf` as an example, you can prepare it as follows:

```shell
huggingface-cli download THUDM/cogvlm-chat-hf --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
huggingface-cli download lmsys/vicuna-7b-v1.5 special_tokens_map.json tokenizer.model tokenizer_config.json --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
```

### Offline inference pipeline

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image


if __name__ == "__main__":
    pipe = pipeline('cogvlm-chat-hf')

    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```
