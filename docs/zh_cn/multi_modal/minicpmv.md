# MiniCPM-V

LMDeploy 支持 MiniCPM-V 系列模型，具体如下：

|        Model         | Supported Inference Engine |
| :------------------: | :------------------------: |
| MiniCPM-Llama3-V-2_5 |         TurboMind          |
|    MiniCPM-V-2_6     |         TurboMind          |

本文将以[MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)为例，演示使用 LMDeploy 部署 MiniCPM-V 系列模型的方法

## 安装

请参考[安装文档](../get_started/installation.md)安装 LMDeploy。

## 离线推理

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('openbmb/MiniCPM-V-2_6')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

更多例子如下：

<details>
  <summary>
    <b>多张图片，多轮对话</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('openbmb/MiniCPM-V-2_6', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(max_slice_nums=9, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image1.jpg')),
        dict(type='image_url', image_url=dict(max_slice_nums=9, url='https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image2.jpg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
print(out.text)

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
print(out.text)
```

</details>

<details>
  <summary>
    <b>上下文小样本学习</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('openbmb/MiniCPM-V-2_6', log_level='INFO')

question = "production date"
messages = [
    dict(role='user', content=[
        dict(type='text', text=question),
        dict(type='image_url', image_url=dict(url='example1.jpg')),
    ]),
    dict(role='assistant', content='2023.08.04'),
    dict(role='user', content=[
        dict(type='text', text=question),
        dict(type='image_url', image_url=dict(url='example2.jpg')),
    ]),
    dict(role='assistant', content='2007.04.24'),
    dict(role='user', content=[
        dict(type='text', text=question),
        dict(type='image_url', image_url=dict(url='test.jpg')),
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
print(out.text)
```

</details>

<details>
  <summary>
    <b>视频对话</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl.utils import encode_image_base64
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord

pipe = pipeline('openbmb/MiniCPM-V-2_6', log_level='INFO')

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number
def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path="video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"

content=[dict(type='text', text=question)]
for frame in frames:
    content.append(dict(type='image_url', image_url=dict(use_image_id=False, max_slice_nums=2,
        url=f'data:image/jpeg;base64,{encode_image_base64(frame)}')))

messages = [dict(role='user', content=content)]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
print(out.text)
```

</details>

## 在线服务

你可以通过 `lmdeploy serve api_server` CLI 工具启动服务：

```shell
lmdeploy serve api_server openbmb/MiniCPM-V-2_6
```

也可以基于 LMDeploy 的 docker 启动服务：

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server openbmb/MiniCPM-V-2_6
```

Docker compose 的方式也是一种选择。在 LMDeploy 代码库的根目录下创建`docker-compose.yml`文件，内容参考如下：

```yaml
version: '3.5'

services:
  lmdeploy:
    container_name: lmdeploy
    image: openmmlab/lmdeploy:latest
    ports:
      - "23333:23333"
    environment:
      HUGGING_FACE_HUB_TOKEN: <secret>
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    stdin_open: true
    tty: true
    ipc: host
    command: lmdeploy serve api_server openbmb/MiniCPM-V-2_6
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
```

然后，你就可以执行命令启动服务了：

```shell
docker-compose up -d
```

通过`docker logs -f lmdeploy`可以查看启动的日志信息，如果发现类似下方的日志信息，就表明服务启动成功了。

```text
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on  http://0.0.0.0:23333  (Press CTRL+C to quit)
```

有关 `lmdeploy serve api_server` 的详细参数可以通过`lmdeploy serve api_server -h`查阅。

关于 `api_server` 更多的介绍，以及访问 `api_server` 的方法，请阅读[此处](api_server_vl.md)
