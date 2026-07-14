# InternLM-XComposer-2.5

## 简介

[InternLM-XComposer-2.5](https://github.com/InternLM/InternLM-XComposer) 是基于书生·浦语2大语言模型研发的突破性的图文多模态大模型，仅使用 7B LLM 后端就达到了 GPT-4V 级别的能力。浦语·灵笔2.5使用24K交错的图像-文本上下文进行训练，通过RoPE外推可以无缝扩展到96K长的上下文。这种长上下文能力使浦语·灵笔2.5在需要广泛输入和输出上下文的任务中表现出色。 LMDeploy 支持了 [internlm/internlm-xcomposer2d5-7b](https://huggingface.co/internlm/internlm-xcomposer2d5-7b) 模型，通过 TurboMind 引擎推理。

## 快速开始

### 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy，并安装上游模型库 InternLM-XComposer-2.5 所需的依赖。

```shell
pip install decord
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

pipe = pipeline('internlm/internlm-xcomposer2d5-7b')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

## Lora 模型

InternLM-XComposer-2.5 针对网页制作和文章创作训练了 LoRA 模型，由于 TurboMind 不支持 slora 特性，所以需要同时只能部署一个 LoRA 模型，需要先对权重进行合并。LMDeploy 提供相关的转换脚本，使用方式为:

```
export HF_MODEL=internlm/internlm-xcomposer2d5-7b
export WORK_DIR=internlm/internlm-xcomposer2d5-7b-web
export TASK=web
python -m lmdeploy.vl.tools.merge_xcomposer2d5_task $HF_MODEL $WORK_DIR --task $TASK
```

## 量化

下面以 base 模型为例，展示量化的方式，若要使用 LoRA 模型，请先按照上一章节提取 LoRA 模型。

```shell

export HF_MODEL=internlm/internlm-xcomposer2d5-7b
export WORK_DIR=internlm/internlm-xcomposer2d5-7b-4bit

lmdeploy lite auto_awq \
   $HF_MODEL \
  --work-dir $WORK_DIR
```

## 更多使用例子

<details>
  <summary>
    <b>Video Understanding</b>
  </summary>

下面以 `pipeline.chat` 为例展示用法，其它接口同样支持推理，需要手动拼接对话内容。

```python
from lmdeploy import pipeline, GenerationConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

HF_MODEL = 'internlm/internlm-xcomposer2d5-7b'
load_video = get_class_from_dynamic_module('ixc_utils.load_video', HF_MODEL)
frame2img = get_class_from_dynamic_module('ixc_utils.frame2img', HF_MODEL)
Video_transform = get_class_from_dynamic_module('ixc_utils.Video_transform', HF_MODEL)
get_font = get_class_from_dynamic_module('ixc_utils.get_font', HF_MODEL)

video = load_video('liuxiang.mp4') # https://github.com/InternLM/InternLM-XComposer/raw/main/examples/liuxiang.mp4
img = frame2img(video, get_font())
img = Video_transform(img)

pipe = pipeline(HF_MODEL)
gen_config = GenerationConfig(top_k=50, top_p=0.8, temperature=1.0)
query = 'Here are some frames of a video. Describe this video in detail'
sess = pipe.chat((query, img), gen_config=gen_config)
print(sess.response.text)

query = 'tell me the athlete code of Liu Xiang'
sess = pipe.chat(query, session=sess, gen_config=gen_config)
print(sess.response.text)
```

</details>

<details>
  <summary>
    <b>Multi-Image</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl import load_image

query = f'Image1 {IMAGE_TOKEN}; Image2 {IMAGE_TOKEN}; Image3 {IMAGE_TOKEN}; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'

urls = ['https://raw.githubusercontent.com/InternLM/InternLM-XComposer/main/examples/cars1.jpg',
        'https://raw.githubusercontent.com/InternLM/InternLM-XComposer/main/examples/cars2.jpg',
        'https://raw.githubusercontent.com/InternLM/InternLM-XComposer/main/examples/cars3.jpg']
images = [load_image(url) for url in urls]

pipe = pipeline('internlm/internlm-xcomposer2d5-7b', log_level='INFO')
output = pipe((query, images), gen_config=GenerationConfig(top_k=0, top_p=0.8, random_seed=89247526689433939))
```

由于 LMDeploy 不支持 beam search，生成的结果与使用 transformers 的 beam search 相比，会有较大的差异，建议关闭 top_k 或者使用较大的 top_k 采样来增加多样性。

</details>

<details>
  <summary>
    <b>Instruction to Webpage</b>
  </summary>

请先使用使用上述说明，转化 web 模型。

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('/nvme/shared/internlm-xcomposer2d5-7b-web', log_level='INFO')
pipe.chat_template.meta_instruction = None

query = 'A website for Research institutions. The name is Shanghai AI lab. Top Navigation Bar is blue.Below left, an image shows the logo of the lab. In the right, there is a passage of text below that describes the mission of the laboratory.There are several images to show the research projects of Shanghai AI lab.'
output = pipe(query, gen_config=GenerationConfig(max_new_tokens=2048))
```

使用 transformers 测试时，发现如果设置了 repetition_penalty，beam search 为1时有较大概率停不下来，因为 LMDeploy 不支持 beam search，建议使用 LMDeploy 推理时关闭 repetition_penalty。

</details>

<details>
  <summary>
    <b>Write Article</b>
  </summary>

请先使用使用上述说明，转化 write 模型。

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('/nvme/shared/internlm-xcomposer2d5-7b-write', log_level='INFO')
pipe.chat_template.meta_instruction = None

query = 'Please write a blog based on the title: French Pastries: A Sweet Indulgence'
output = pipe(query, gen_config=GenerationConfig(max_new_tokens=8192))
```

</details>
