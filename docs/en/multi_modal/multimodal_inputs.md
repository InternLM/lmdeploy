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

Image loading and encoding apply valid EXIF orientation before mode conversion or processing, including direct PIL image inputs. Applied orientation tags are removed so processing the result again does not rotate it twice; the caller's PIL image is not modified. Images without EXIF orientation retain their existing behavior. Invalid orientation metadata is ignored without hiding pixel-decoding errors. If you rotate pixels yourself, remove or reset the old orientation tag before passing the image to LMDeploy.

This corrects the previous behavior for EXIF orientations 2–8 across models; it does not change pixel/token budget units or video frame sampling. Base64 JPEG video frames use the same image decoder and also inherit orientation normalization; this does not establish training/serving video preprocessing equivalence.

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

## Images in Tool Results

Chat `tool` messages, Messages `tool_result` blocks, and Responses `function_call_output` items preserve images together with their result's call ID and text. Support also depends on the model's chat template and vision preprocessor; ordinary image support alone does not guarantee tool-image support.

For GLM-5.3-Flash's default template, preprocessing aligns unambiguous tool-result groups with the assistant's call order before both rendering and image collection. This ordering rule is limited to `Glm5NextForConditionalGeneration`; other architectures retain their input order. It does not infer ordering rules from arbitrary or customized Jinja templates. InternVL3 and InternVL3.5 normalize both user and tool images while retaining tool metadata, including when legacy `<IMAGE_TOKEN>` placeholders are present in another message.

See the [Messages API](../llm/api_server_anthropic.md) and [Responses API](../llm/api_server_responses.md) documentation for their tool-result formats.

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

> **Note:** Native video input is currently supported for **Qwen3-VL**, **Qwen3.5**, **Qwen3-Omni**, **InternS1-Pro**, and **Intern-S2-Preview** models only.

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

> **Note:** Native video input is currently supported for **Qwen3-VL**, **Qwen3.5**, **Qwen3-Omni**, **InternS1-Pro**, and **Intern-S2-Preview** models only.

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

## Mixed Image and Video

> **Note:** Native mixed image/video input is currently supported for **Qwen3-VL**, **Qwen3.5**, **Qwen3-Omni**, **InternS1-Pro**, and **Intern-S2-Preview** models only.

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

## Single Audio

> **Note:** Audio input is currently supported for **Qwen3-Omni** models only.

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

> **Note:** Audio input is currently supported for **Qwen3-Omni** models only.

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

When the API server is exposed to untrusted clients, use `--allowed-media-domains` at startup to restrict which hostnames HTTP(S) media URLs can fetch from and reduce SSRF risk. The option matches exact hostnames; for example, `--allowed-media-domains example.com` allows `https://example.com/...` but not `https://cdn.example.com/...`.

```shell
lmdeploy serve api_server <model_path> --allowed-media-domains example.com
```

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
<summary>Base64 encoding example (audio)</summary>

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_audio_base64

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')
model_name = client.models.list().data[0].id

b64 = encode_audio_base64('/path/to/your/audio.wav')
audio_url = f'data:audio/wav;base64,{b64}'

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'audio_url',
                'audio_url': {'url': audio_url},
            },
            {'type': 'text', 'text': 'Describe this audio.'},
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

- **`mm_processor_kwargs`**: passes model-specific HF processor options, keyed by `image`, `video`, or `audio`
- **`media_io_kwargs`**: controls how media is loaded (e.g. video frame sampling rate/count and audio sampling rate)

Both are passed as extra fields in the API request body via `extra_body`, or directly to `pipe()` when using the pipeline API.

`/generate`, `/v1/chat/completions`, and the Anthropic-compatible `/v1/messages` endpoint accept these fields at the top level of the JSON body. SDK `extra_body` merges them into that body.

For native GLM-5.3-Flash processors, use **token budgets**, not pixel budgets:

```json
{"mm_processor_kwargs": {"image": {"min_image_tokens": 64, "max_image_tokens": 64}}}
```

The same token-budget names can control video spatial processing under `video`. Overrides are request-local and do not change the shared processor defaults. A single bound can be supplied; the other bound uses the processor default. Bounds must be positive integers with minimum no greater than maximum.

Native GLM processors do not use the generic `size` contract: `min_pixels`, `max_pixels`, and `size` are rejected rather than silently ignored. Qwen and the older GLM-4V compatibility processor retain the pixel-budget syntax shown below; the compatibility processor does not support native GLM token-budget overrides. XTuner's GLM tokenizer currently names its token-valued configuration `min_pixels/max_pixels`; map those values to the explicit `min_image_tokens/max_image_tokens` request keys instead of copying the names.

### Video sampling and backward compatibility

For video files and URLs, LMDeploy's media loader decodes and samples frames **before** calling the HF processor. Set `media_io_kwargs.video.fps/num_frames` to control sampling. The loader defaults to at most 32 frames; when both a positive `fps` and `num_frames` apply, the tighter frame-count limit wins. The processor's own sampling defaults do not automatically configure the loader.

To preserve older requests, `fps`, `num_frames`, `min_frames`, `max_frames`, `do_sample_frames`, and `video_metadata` inside `mm_processor_kwargs.video` are **ignored with a warning**, not rejected or moved to `media_io_kwargs`. HF frame resampling remains disabled, and metadata comes from the media loader. `min_frames/max_frames` do not currently have matching loader options.

For size-based processors such as Qwen and InternS2, the existing `min_pixels/max_pixels` behavior is retained: pixel aliases take precedence over a simultaneously supplied `size`, and an inverted range falls back to processor defaults. Other supported HF options now take effect when explicitly supplied. The native GLM pixel-field rejection described above is a separate compatibility change; it does not apply to size-based processors.

If the caller has already sampled frames and sends a `video/jpeg` sequence, LMDeploy keeps all supplied frames rather than applying the ordinary loader's frame-count limit again. In this path, `num_frames` does not trim the sequence; `fps` is used to construct fallback metadata, not to select frames.

These rules define serving behavior, not training/serving equivalence. Matching parameter values alone does not guarantee identical frame indices or timestamps across XTuner and LMDeploy; their sampling algorithms and metadata must also be checked.

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

if __name__ == '__main__':
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
