# VL-LLM Offline Inference Pipeline

In this tutorial, We will present a list of examples to introduce the usage of `lmdeploy.vl.pipeline`.

The usage of `lmdeploy.vl.pipeline` is similar to `lmdeploy.pipeline`. You can find the detailed parameter description in [this](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html) guide.

## Usage

- **An example using default parameters:**

```python
from lmdeploy.vl import pipeline, load_image_from_url
from lmdeploy import TurbomindEngineConfig, ChatTemplateConfig

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
    backend_config=TurbomindEngineConfig(session_len=8192),
    chat_template_config=ChatTemplateConfig(model_name='vicuna'))

image = load_image_from_url('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
response = pipe(('describe this image', [image]))
print(response)
```

- **An example for OpenAI format prompt input:**

```python
from lmdeploy.vl import pipeline
from lmdeploy import TurbomindEngineConfig, ChatTemplateConfig

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
    backend_config=TurbomindEngineConfig(session_len=8192),
    chat_template_config=ChatTemplateConfig(model_name='vicuna'))

prompts = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg"}}
        ]
    }
]


response = pipe(prompts)
print(response)
```

- **An example for streaming mode:**

```python
from lmdeploy.vl import pipeline
from lmdeploy import  GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig

backend_config = TurbomindEngineConfig(session_len=8192)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
chat_template_config=ChatTemplateConfig(model_name='vicuna')
pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=backend_config,
                chat_template_config=chat_template_config)
prompts = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg"}}
        ]
    }
]

for item in pipe.stream_infer(prompts, gen_config=gen_config):
    print(item.text, end='', flush=True)
```
