# InternLM-XComposer-2.5

## Introduction

[InternLM-XComposer-2.5](https://github.com/InternLM/InternLM-XComposer) excels in various text-image comprehension and composition applications, achieving GPT-4V level capabilities with merely 7B LLM backend. IXC-2.5 is trained with 24K interleaved image-text contexts, it can seamlessly extend to 96K long contexts via RoPE extrapolation. This long-context capability allows IXC-2.5 to perform exceptionally well in tasks requiring extensive input and output contexts. LMDeploy supports model [internlm/internlm-xcomposer2d5-7b](https://huggingface.co/internlm/internlm-xcomposer2d5-7b)  in TurboMind engine.

## Quick Start

### Installation

Install LMDeploy with pip (Python 3.8+). Refer to [Installation](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation) for more.

```shell
pip install lmdeploy

# install other packages that InternLM-XComposer-2.5 needs
pip install decord
```

### Offline inference pipeline

The following sample code shows the basic usage of VLM pipeline. For more examples, please refer to [VLM Offline Inference Pipeline](https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#vlm-offline-inference-pipeline)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

pipe = pipeline('internlm/internlm-xcomposer2d5-7b')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

## Lora Model

InternLM-XComposer-2.5 trained the LoRA weights for webpage creation and article writing. As TurboMind backend doesn't support slora, only one LoRA model can be deployed at a time, and the LoRA weights need to be merged when deploying the model. LMDeploy provides the corresponding conversion script, which is used as follows:

```
export HF_MODEL=internlm/internlm-xcomposer2d5-7b
export WORK_DIR=internlm/internlm-xcomposer2d5-7b-web
export TASK=web
python -m lmdeploy.vl.tools.merge_xcomposer2d5_task $HF_MODEL $WORK_DIR --task $TASK
```

## Quantization

The following takes the base model as an example to show the quantization method. If you want to use the LoRA model, please merge the LoRA model according to the previous section.

```shell

export HF_MODEL=internlm/internlm-xcomposer2d5-7b
export WORK_DIR=internlm/internlm-xcomposer2d5-7b-4bit

lmdeploy lite auto_awq \
   $HF_MODEL \
  --work-dir $WORK_DIR
```

## More examples

<details>
  <summary>
    <b>Video Understanding</b>
  </summary>

The following uses the `pipeline.chat` interface api as an example to demonstrate its usage. Other interfaces apis also support inference but require manually splicing of conversation content.

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

Since LMDeploy does not support beam search, the generated results will be quite different from those using beam search with transformers. It is recommended to turn off top_k or use a larger top_k sampling to increase diversity.

</details>

<details>
  <summary>
    <b>Instruction to Webpage</b>
  </summary>

Please first convert the web model using the instructions above.

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('/nvme/shared/internlm-xcomposer2d5-7b-web', log_level='INFO')
pipe.chat_template.meta_instruction = None

query = 'A website for Research institutions. The name is Shanghai AI lab. Top Navigation Bar is blue.Below left, an image shows the logo of the lab. In the right, there is a passage of text below that describes the mission of the laboratory.There are several images to show the research projects of Shanghai AI lab.'
output = pipe(query, gen_config=GenerationConfig(max_new_tokens=2048))
```

When using transformers for testing, it is found that if repetition_penalty is set, there is a high probability that the decode phase will not stop if `num_beams` is set to 1. As LMDeploy does not support beam search, it is recommended to turn off repetition_penalty when using LMDeploy for inference.

</details>

<details>
  <summary>
    <b>Write Article</b>
  </summary>

Please first convert the write model using the instructions above.

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('/nvme/shared/internlm-xcomposer2d5-7b-write', log_level='INFO')
pipe.chat_template.meta_instruction = None

query = 'Please write a blog based on the title: French Pastries: A Sweet Indulgence'
output = pipe(query, gen_config=GenerationConfig(max_new_tokens=8192))
```

</details>
