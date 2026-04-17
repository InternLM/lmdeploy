# Multi-Modal Inputs

LMDeploy uses the OpenAI message format for all modalities. Each content item in a message is a dict with a `type` field that determines how it is decoded.

**Quick reference:**

| Modality    | `type` key        | URL field             |
| ----------- | ----------------- | --------------------- |
| Text        | `text`            | —                     |
| Image       | `image_url`       | `image_url.url`       |
| Video       | `video_url`       | `video_url.url`       |
| Audio       | `audio_url`       | `audio_url.url`       |
| Time Series | `time_series_url` | `time_series_url.url` |

All examples below target the lmdeploy OpenAI-compatible API server. Start it with:

```bash
lmdeploy serve api_server <model_path> --server-port 23333
```

______________________________________________________________________

## Text

<details>
<summary>Complete example</summary>

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
            'text': 'Who are you?',
        }],
    }],
    temperature=0.8,
    top_p=0.8,
)
print(response.choices[0].message.content)
```

</details>

______________________________________________________________________

## Single Image

<details>
<summary>Complete example</summary>

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
                'text': 'Describe this image.',
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

## Multiple Images

<details>
<summary>Complete example</summary>

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
                'text': 'Compare these two images. What are the similarities and differences?',
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

## Single Video

> **Note:** Native video input is currently supported for **Qwen3-VL**, **Qwen3.5**, and **InternS1-Pro** models only.

<details>
<summary>Complete example</summary>

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
                'text': "What's in this video?",
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

## Multiple Videos

> **Note:** Native video input is currently supported for **Qwen3-VL**, **Qwen3.5**, and **InternS1-Pro** models only.

<details>
<summary>Complete example</summary>

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
                'text': 'Compare these two videos. What are the similarities and differences?',
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

## Single Audio

<details>
<summary>Complete example</summary>

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
                'type': 'audio_url',
                'audio_url': {
                    'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav',
                },
            },
            {
                'type': 'text',
                'text': 'Describe this audio.',
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

## Multiple Audios

<details>
<summary>Complete example</summary>

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

audio_url_1 = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav'
audio_url_2 = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'audio_url', 'audio_url': {'url': audio_url_1}},
            {'type': 'audio_url', 'audio_url': {'url': audio_url_2}},
            {
                'type': 'text',
                'text': 'Compare these two audios. What are the similarities and differences?',
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

## Mixed Image and Video

> **Note:** Native video input is currently supported for **Qwen3-VL**, **Qwen3.5**, and **InternS1-Pro** models only.

<details>
<summary>Complete example</summary>

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
                'text': 'Describe both the image and the video.',
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

## Time Series

> **Note:** Time series input is currently supported for the **InternS1-Pro** model only.

The `time_series_url` content item requires a `sampling_rate` field (in Hz) alongside the URL.

<details>
<summary>Complete example</summary>

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
                'text': ('Please determine whether an Earthquake event has occurred. '
                         'If so, specify P-wave and S-wave starting indices.'),
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

## Local Files and Base64

In addition to HTTP URLs, lmdeploy accepts:

- **Local file paths** via `file://` scheme: `file:///absolute/path/to/file.jpg`
- **Base64-encoded data** via data URLs: `data:<mime>;base64,<encoded_data>`

Use the helpers in `lmdeploy.vl.utils` to encode local files:

<details>
<summary>Local file path example</summary>

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
            {'type': 'text', 'text': 'Describe this image.'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 encoding example (image)</summary>

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
            {'type': 'text', 'text': 'Describe this image.'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 encoding example (video)</summary>

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_video_base64

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

# num_frames controls how many frames to sample before encoding
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
            {'type': 'text', 'text': 'Describe this video.'},
        ],
    }],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary>Base64 encoding example (time series)</summary>

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
            {'type': 'text', 'text': 'Analyze this time series.'},
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

## Processor and IO kwargs

Two optional parameters let you control media processing:

- **`mm_processor_kwargs`**: controls vision token resolution (min/max pixels per image or video frame)
- **`media_io_kwargs`**: controls how media is loaded (e.g. video frame sampling rate and count)

Both are passed as extra fields in the API request body via `extra_body`, or directly to `pipe()` when using the pipeline API.

<details>
<summary>API server example (extra_body)</summary>

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
            {'type': 'text', 'text': 'Describe this video.'},
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
<summary>Pipeline API equivalent</summary>

```python
from lmdeploy import pipeline, PytorchEngineConfig

pipe = pipeline('<model_path>', backend_config=PytorchEngineConfig(tp=1))

video_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'

messages = [{
    'role': 'user',
    'content': [
        {'type': 'video_url', 'video_url': {'url': video_url}},
        {'type': 'text', 'text': 'Describe this video.'},
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
