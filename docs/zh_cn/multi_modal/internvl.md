# InternVL2

## 简介

InternVL是一个开源的视觉语言基础模型，它将Vision Transformer（ViT）扩展至6亿参数，并与大型语言模型（LLM）对齐。作为目前最大的开源视觉/视觉语言基础模型（14亿参数），InternVL在视觉感知、跨模态检索、多模态对话等多个任务上实现了32项最先进的性能。LMDeploy 支持了 InternVL 系列模型的推理。下面以 InternVL2-8B 为例，展示其使用方法。

## 快速开始

### 安装

请参考[安装文档](../installation.md)安装 LMDeploy，并安装上游模型库 InternVL2 所需的依赖。

```shell
pip install timm
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL2-8B')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

## 更多使用例子

<details>
  <summary>
    <b>多图多轮对话，拼接图像</b>
  </summary>

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.utils import encode_image_base64
from lmdeploy.vl.constants import IMAGE_TOKEN

pipe = pipeline('OpenGVLab/InternVL2-8B', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text=f'<img>{IMAGE_TOKEN}{IMAGE_TOKEN}</img>\nDescribe the two images in detail.'),
        dict(type='image_url', image_url=dict(max_dynamic_patch=12, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image1.jpg')),
        dict(type='image_url', image_url=dict(max_dynamic_patch=12, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image2.jpg'))
    ])
]
out = pipe(messages, top_k=1)

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, top_k=1)
```

</details>

<details>
  <summary>
    <b>多图多轮对话，独立图像</b>
  </summary>

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.utils import encode_image_base64
from lmdeploy.vl.constants import IMAGE_TOKEN

pipe = pipeline('OpenGVLab/InternVL2-8B', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text=f'Image-1: <img>{IMAGE_TOKEN}</img>\nImage-2: <img>{IMAGE_TOKEN}</img>\nDescribe the two images in detail.'),
        dict(type='image_url', image_url=dict(max_dynamic_patch=12, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image1.jpg')),
        dict(type='image_url', image_url=dict(max_dynamic_patch=12, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image2.jpg'))
    ])
]
out = pipe(messages, top_k=1)

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, top_k=1)
```

</details>

<details>
  <summary>
    <b>视频多轮对话</b>
  </summary>

```python
import numpy as np
from lmdeploy import pipeline
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
pipe = pipeline('OpenGVLab/InternVL2-8B', log_level='INFO')


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs


video_path = 'red-panda.mp4'
imgs = load_video(video_path, num_segments=8, max_num=1)

question = ''
for i in range(len(imgs)):
    question = question + f'Frame{i+1}: <img>{IMAGE_TOKEN}</img>\n'

question += 'What is the red panda doing?'

content = [{'type': 'text', 'text': question}]
for img in imgs:
    content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

messages = [dict(role='user', content=content)]
out = pipe(messages, top_k=1)

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='Describe this video in detail. Don\'t repeat.'))
out = pipe(messages, top_k=1)
```

</details>
