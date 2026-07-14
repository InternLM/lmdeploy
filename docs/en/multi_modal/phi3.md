# Phi-3 Vision

## Introduction

[Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) is a family of small language and multi-modal models from MicroSoft. LMDeploy supports the multi-modal models as below.

|                                                Model                                                | Size | Supported Inference Engine |
| :-------------------------------------------------------------------------------------------------: | :--: | :------------------------: |
| [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | 4.2B |          PyTorch           |
|    [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)    | 4.2B |          PyTorch           |

The next chapter demonstrates how to deploy an Phi-3 model using LMDeploy, with [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) as an example.

## Installation

Please install LMDeploy by following the [installation guide](../get_started/installation.md) and install the dependency [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

```shell
# It is recommended to find the whl package that matches the environment from the releases on https://github.com/Dao-AILab/flash-attention.
pip install flash-attn
```

## Offline inference

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('microsoft/Phi-3.5-vision-instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## Online serving

### Launch Service

You can launch the server by the `lmdeploy serve api_server` CLI:

```shell
lmdeploy serve api_server microsoft/Phi-3.5-vision-instruct
```

### Integrate with `OpenAI`

Here is an example of interaction with the endpoint `v1/chat/completions` service via the openai package.
Before running it, please install the openai package by `pip install openai`

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
