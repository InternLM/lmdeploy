# 多模态输入

LMDeploy 使用 OpenAI 消息格式处理所有模态。消息中的每个内容项都是一个包含 `type` 字段的字典，该字段决定了数据的解码方式。

**快速参考：**

| 模态     | `type` 字段       | URL 字段              |
| -------- | ----------------- | --------------------- |
| 文本     | `text`            | —                     |
| 图像     | `image_url`       | `image_url.url`       |
| 视频     | `video_url`       | `video_url.url`       |
| 时序数据 | `time_series_url` | `time_series_url.url` |

以下示例均面向 lmdeploy 兼容 OpenAI 的 API 服务。启动服务：

```bash
lmdeploy serve api_server <model_path> --server-port 23333
```

______________________________________________________________________

## 纯文本

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': '你是谁？',
        }],
    }],
    temperature=0.8,
    top_p=0.8,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 单张图像

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
                },
            },
            {
                'type': 'text',
                'text': '描述这张图片。',
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 多张图像

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
                },
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
                },
            },
            {
                'type': 'text',
                'text': '比较这两张图片，有哪些相似点和不同点？',
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 单个视频

> **注意：** 原生视频输入目前仅支持 **Qwen3-VL**、**Qwen3.5** 和 **InternS1-Pro** 模型。

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

video_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'video_url',
                'video_url': {
                    'url': video_url,
                },
            },
            {
                'type': 'text',
                'text': '这个视频里有什么？',
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 多个视频

> **注意：** 原生视频输入目前仅支持 **Qwen3-VL**、**Qwen3.5** 和 **InternS1-Pro** 模型。

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

video_url_1 = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'
video_url_2 = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'video_url',
                'video_url': {'url': video_url_1},
            },
            {
                'type': 'video_url',
                'video_url': {'url': video_url_2},
            },
            {
                'type': 'text',
                'text': '比较这两个视频，有哪些相似点和不同点？',
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 图像与视频混合

> **注意：** 原生视频输入目前仅支持 **Qwen3-VL**、**Qwen3.5** 和 **InternS1-Pro** 模型。

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

image_url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
video_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {'url': image_url},
            },
            {
                'type': 'video_url',
                'video_url': {'url': video_url},
            },
            {
                'type': 'text',
                'text': '描述这张图片和这个视频。',
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 时序数据

> **注意：** 时序数据输入目前仅支持 **InternS1-Pro** 模型。

`time_series_url` 内容项需要在 URL 之外额外提供 `sampling_rate` 字段（单位：Hz）。

<details>
<summary>完整示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': '请判断是否发生了地震事件，若有请指出P波和S波的起始索引。',
            },
            {
                'type': 'time_series_url',
                'time_series_url': {
                    'url': 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0092638_seism.npy',
                    'sampling_rate': 100,
                },
            },
        ],
    }],
    temperature=0.8,
    top_p=0.8,
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 本地文件与 Base64

除 HTTP URL 外，lmdeploy 还支持：

- **本地文件路径**，使用 `file://` 协议：`file:///absolute/path/to/file.jpg`
- **Base64 编码数据**，使用 data URL：`data:<mime>;base64,<encoded_data>`

可使用 `lmdeploy.vl.utils` 中的工具函数对本地文件进行编码：

<details>
<summary>本地文件路径示例</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'file:///path/to/your/image.jpg',
                },
            },
            {'type': 'text', 'text': '描述这张图片。'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 编码示例（图像）</summary>

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_image_base64

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

b64 = encode_image_base64('/path/to/your/image.jpg')
image_url = f'data:image/jpeg;base64,{b64}'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {'url': image_url},
            },
            {'type': 'text', 'text': '描述这张图片。'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 编码示例（视频）</summary>

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_video_base64

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

# num_frames 控制编码前采样的帧数
b64 = encode_video_base64('/path/to/your/video.mp4', num_frames=16)
video_url = f'data:video/mp4;base64,{b64}'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'video_url',
                'video_url': {'url': video_url},
            },
            {'type': 'text', 'text': '描述这个视频。'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 编码示例（时序数据）</summary>

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_time_series_base64

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

b64 = encode_time_series_base64('/path/to/your/data.npy')
ts_url = f'data:application/octet-stream;base64,{b64}'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': '分析这段时序数据。'},
            {
                'type': 'time_series_url',
                'time_series_url': {
                    'url': ts_url,
                    'sampling_rate': 100,
                },
            },
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## 处理器与 IO 参数

两个可选参数用于控制媒体处理行为：

- **`mm_processor_kwargs`**：控制视觉 token 的分辨率（每张图片或视频帧的最小/最大像素数）
- **`media_io_kwargs`**：控制媒体加载方式（如视频帧采样率和帧数）

两者均通过 `extra_body` 作为请求体中的额外字段传入 API，或在使用 pipeline API 时直接传给 `pipe()`。

<details>
<summary>API 服务示例（extra_body）</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

video_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'video_url', 'video_url': {'url': video_url}},
            {'type': 'text', 'text': '描述这个视频。'},
        ],
    }],
    max_completion_tokens=256,
    extra_body={
        'mm_processor_kwargs': {
            'video': {
                'min_pixels': 4 * 32 * 32,
                'max_pixels': 256 * 32 * 32,
            },
        },
        'media_io_kwargs': {
            'video': {
                'num_frames': 16,
                'fps': 2,
            },
        },
    },
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Pipeline API 等价写法</summary>

```python
from lmdeploy import pipeline, PytorchEngineConfig

pipe = pipeline('<model_path>', backend_config=PytorchEngineConfig(tp=1))

video_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

messages = [{
    'role': 'user',
    'content': [
        {'type': 'video_url', 'video_url': {'url': video_url}},
        {'type': 'text', 'text': '描述这个视频。'},
    ],
}]

response = pipe(
    messages,
    mm_processor_kwargs={
        'video': {
            'min_pixels': 4 * 32 * 32,
            'max_pixels': 256 * 32 * 32,
        },
    },
    media_io_kwargs={
        'video': {
            'num_frames': 16,
            'fps': 2,
        },
    },
)
print(response)
```

</details>
