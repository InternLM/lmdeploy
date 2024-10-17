# Mllama

## Introduction

[Llama3.2-VL](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) is a family of large language and multi-modal models from Meta.

We will demonstrate how to deploy an Llama3.2-VL model using LMDeploy, with [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) as an example.

## Installation

Please install LMDeploy by following the [installation guide](../get_started/installation.md).

## Offline inference

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('meta-llama/Llama-3.2-11B-Vision-Instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## Online serving

### Launch Service

You can launch the server by the `lmdeploy serve api_server` CLI:

```shell
lmdeploy serve api_server meta-llama/Llama-3.2-11B-Vision-Instruct
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
