# VL-LLM 离线推理 pipeline

视觉推理 pipeline 与 文本推理 pipeline 的用法大体相同，输入略有区别。下面通过一些例子展示其用法，具体参数设置可参考[文本推理 pipeline](https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html)

## 使用方法

- **使用默认参数的例子:**

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

- **如何设置 OpenAI 格式输入:**

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

- **流式返回处理结果：**

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
