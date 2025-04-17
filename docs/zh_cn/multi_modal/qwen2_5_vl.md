# Qwen2.5-VL

LMDeploy 支持 Qwen-VL 系列模型，具体如下：

|   Model    |       Size       | Supported Inference Engine |
| :--------: | :--------------: | :------------------------: |
| Qwen2.5-VL | 3B, 7B, 32B, 72B |          PyTorch           |

本文将以[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)为例，演示使用 LMDeploy 部署 Qwen2.5-VL 系列模型的方法

## 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy，并安装上游 Qwen2.5-VL 模型库所需的依赖。

```shell
# Qwen2.5-VL requires the latest transformers (transformers >= 4.49.0)
pip install git+https://github.com/huggingface/transformers
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]==0.0.8
```

## 离线推理

以下是使用 pipeline 进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('Qwen/Qwen2.5-VL-7B-Instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

更多例子如下：

<details>
  <summary>
    <b>多图多轮对话</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('Qwen/Qwen2.5-VL-7B-Instruct', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
```

</details>

<details>
  <summary>
    <b>控制图片分辨率，加速推理</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('Qwen/Qwen2.5-VL-7B-Instruct', log_level='INFO')

min_pixels = 64 * 28 * 28
max_pixels = 64 * 28 * 28
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(min_pixels=min_pixels, max_pixels=max_pixels, url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(min_pixels=min_pixels, max_pixels=max_pixels, url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
```

</details>

<details>
  <summary>
    <b>视频多轮对话</b>
  </summary>

```python
import numpy as np
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
pipe = pipeline('Qwen/Qwen2.5-VL-7B-Instruct', log_level='INFO')


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


def load_video(video_path, bound=None, num_segments=32):
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
imgs = load_video(video_path, num_segments=8)

question = ''
for i in range(len(imgs)):
    question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

question += 'What is the red panda doing?'

content = [{'type': 'text', 'text': question}]
for img in imgs:
    content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

messages = [dict(role='user', content=content)]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='Describe this video in detail. Don\'t repeat.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
```

</details>
