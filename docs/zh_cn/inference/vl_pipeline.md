# VLM 离线推理 pipeline

LMDeploy 把视觉-语言模型（VLM）复杂的推理过程，抽象为简单好用的 pipeline。它的用法与大语言模型（LLM）推理 [pipeline](./pipeline.md) 类似。本文将以 [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) 模型为例，通过若干示例，展示 VLM pipeline 的强大能力。​
首先，我们会展示 pipeline 最基础的用法，并在此基础上，通过引擎的配置和生成条件配置，逐步引出更多能力，比如模型并行、自定义上下文长度、随机采样等等。然后，针对多图、批量提示词等场景，给出对应的推理示例。

## "Hello, world" 示例

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

如果在执行这个用例时，出现 `ImportError` 的错误，请按照提示安装相关的依赖包。​
​
上面的例子中，推理时的提示词是 (prompt, image) 的 tuple 结构。除了这种结构外，pipeline 支持 openai 格式的提示词：

```python
from lmdeploy import pipeline

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b')

prompts = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'describe this image'},
            {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
        ]
    }
]
response = pipe(prompts)
print(response)
```

### 设置多卡并行

设置引擎参数 `tp`，可激活多卡并行能力

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=TurbomindEngineConfig(tp=2))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

### 设置上下文长度

创建 pipeline 时，通过设置引擎参数 `session_len`，可以定制上下文窗口的最大长度

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

### 设置随机采样参数

pipeline 的生成接口，默认使用 greedy search 采样方式。需要引入随机采样时，可传入 `top_k`、`top_p` 和 `temperature` 等参数

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=TurbomindEngineConfig(tp=2, session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.6)
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image), gen_config=gen_config)
print(response)
```

### 设置对话模板

推理时，LMDeploy 会根据模型路径匹配内置的对话模板，并把对话模板应用到输入的提示词上。但是，对于类似 [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) 视觉-语言模型，它使用的对话模板是 vicuna，但是这个模板名无法从模型路径中获取，所以需要用户指定。具体方式如下：

```python
from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
pipe = pipeline('liuhaotian/llava-v1.5-7b',
                chat_template_config=ChatTemplateConfig(model_name='vicuna'))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

关于如何自定义对话模版，请参考[这里](../advance/chat_template.md)

## 多图推理

对于多图的场景，在推理时，只要把它们放在一个列表中即可。不过，多图意味着输入 token 数更多，所以通常需要[增大推理的上下文长度](#设置上下文长度​)。

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
response = pipe(('describe these images', images))
print(response)
```

## 提示词批处理

做批量提示词推理非常简单，只要把它们放在一个 list 结构中：

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b',
                backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```
