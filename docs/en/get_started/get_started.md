# Quick Start

This tutorial shows the usage of LMDeploy on CUDA platform:

- Offline inference of LLM model and VLM model
- Serve a LLM or VLM model by the OpenAI compatible server
- Console CLI to interactively chat with LLM model

Before reading further, please ensure that you have installed lmdeploy as outlined in the [installation guide](installation.md)

## Offline batch inference

### LLM inference

```python
from lmdeploy import pipeline
pipe = pipeline('internlm/internlm2_5-7b-chat')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

When constructing the `pipeline`, if an inference engine is not designated between the TurboMind Engine and the PyTorch Engine, LMDeploy will automatically assign one based on [their respective capabilities](../supported_models/supported_models.md), with the TurboMind Engine taking precedence by default.

However, you have the option to manually select an engine. For instance,

```python
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=TurbomindEngineConfig(
                    max_batch_size=32,
                    enable_prefix_caching=True,
                    cache_max_entry_count=0.8,
                    session_len=8192,
                ))
```

or,

```python
from lmdeploy import pipeline, PytorchEngineConfig
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=PytorchEngineConfig(
                    max_batch_size=32,
                    enable_prefix_caching=True,
                    cache_max_entry_count=0.8,
                    session_len=8192,
                ))
```

```{note}
The parameter "cache_max_entry_count" significantly influences the GPU memory usage.
It means the proportion of FREE GPU memory occupied by the K/V cache after the model weights are loaded.

The default value is 0.8. The K/V cache memory is allocated once and reused repeatedly, which is why it is observed that the built pipeline and the "api_server" mentioned later in the next consumes a substantial amount of GPU memory.

If you encounter an Out-of-Memory(OOM) error, you may need to consider lowering the value of "cache_max_entry_count".
```

When use the callable `pipe()` to perform token generation with given prompts, you can set the sampling parameters via `GenerationConfig` as below:

```python
from lmdeploy import GenerationConfig, pipeline

pipe = pipeline('internlm/internlm2_5-7b-chat')
prompts = ['Hi, pls intro yourself', 'Shanghai is']
response = pipe(prompts,
                gen_config=GenerationConfig(
                    max_new_tokens=1024,
                    top_p=0.8,
                    top_k=40,
                    temperature=0.6
                ))
```

In the `GenerationConfig`, `top_k=1` or `temperature=0.0` indicates greedy search.

For more information about pipeline, please read the [detailed tutorial](../llm/pipeline.md)

### VLM inference

The usage of VLM inference pipeline is akin to that of LLMs, with the additional capability of processing image data with the pipeline.
For example, you can utilize the following code snippet to perform the inference with an InternVL model:

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL2-8B')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

In VLM pipeline, the default image processing batch size is 1. This can be adjusted by `VisionConfig`. For instance, you might set it like this:

```python
from lmdeploy import pipeline, VisionConfig
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL2-8B',
                vision_config=VisionConfig(
                    max_batch_size=8
                ))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

However, the larger the image batch size, the greater risk of an OOM error, because the LLM component within the VLM model pre-allocates a massive amount of memory in advance.

We encourage you to manually choose between the TurboMind Engine and the PyTorch Engine based on their respective capabilities, as detailed in [the supported-models matrix](../supported_models/supported_models.md).
Additionally, follow the instructions in [LLM Inference](#llm-inference) section to reduce the values of memory-related parameters

## Serving

As demonstrated in the previous [offline batch inference](#offline-batch-inference) section, this part presents the respective serving methods for LLMs and VLMs.

### Serve a LLM model

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

This command will launch an OpenAI-compatible server on the localhost at port `23333`. You can specify a different server port by using the `--server-port` option.
For more options, consult the help documentation by running `lmdeploy serve api_server --help`. Most of these options align with the engine configuration.

To access the service, you can utilize the official OpenAI Python package `pip install openai`. Below is an example demonstrating how to use the entrypoint `v1/chat/completions`

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": " provide three suggestions about time management"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

We encourage you to refer to the detailed guide for more comprehensive information about [serving with Docker](../llm/api_server.md), [function calls](../llm/api_server_tools.md) and other topics

### Serve a VLM model

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B
```

```{note}
LMDeploy reuses the vision component from upstream VLM repositories. Each upstream VLM model may have different dependencies.
Consequently, LMDeploy has decided not to include the dependencies of the upstream VLM repositories in its own dependency list.
If you encounter an "ImportError" when using LMDeploy for inference with VLM models, please install the relevant dependencies yourself.
```

After the service is launched successfully, you can access the VLM service in a manner similar to how you would access the `gptv4` service by modifying the `api_key` and `base_url` parameters:

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

## Inference with Command line Interface

LMDeploy offers a very convenient CLI tool for users to chat with the LLM model locally. For example:

```shell
lmdeploy chat internlm/internlm2_5-7b-chat --backend turbomind
```

It is designed to assist users in checking and verifying whether LMDeploy supports their model, whether the chat template is applied correctly, and whether the inference results are delivered smoothly.

Another tool, `lmdeploy check_env`, aims to gather the essential environment information. It is crucial when reporting an issue to us, as it helps us diagnose and resolve the problem more effectively.

If you have any doubt about their usage, you can try using the `--help` option to obtain detailed information.
