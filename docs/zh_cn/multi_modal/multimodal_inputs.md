# 多模态输入

LMDeploy 使用 OpenAI 消息格式处理所有模态。消息中的每个内容项都是一个包含 `type` 字段的字典，该字段决定了数据的解码方式。

**快速参考：**

| 模态     | `type` 字段       | URL 字段              |
| -------- | ----------------- | --------------------- |
| 文本     | `text`            | —                     |
| 图像     | `image_url`       | `image_url.url`       |
| 视频     | `video_url`       | `video_url.url`       |
| 音频     | `audio_url`       | `audio_url.url`       |
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

图片加载和编码会在模式转换或处理前应用有效的 EXIF 方向，包括直接传入的 PIL 图片。已应用的方向标签会被移除，避免再次处理时重复旋转；调用者的 PIL 图片不会被修改。没有 EXIF 方向的图片保持原有行为。无效的方向元数据会被忽略，但不会掩盖像素解码错误。如果自行旋转了图片像素，请在传给 LMDeploy 前移除或重置旧的方向标签。

这会纠正各模型在 EXIF 方向为 2–8 时的旧行为，但不改变像素/token 预算单位或视频抽帧策略。Base64 JPEG 视频帧复用同一图片解码器，也会应用方向归一化；这不代表训练侧与服务侧的视频预处理已完全对齐。

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

## 工具结果中的图像

Chat 的 `tool` 消息、Messages 的 `tool_result` 块和 Responses 的 `function_call_output` 项会将图片与对应结果的调用 ID、文本保留在一起。模型的聊天模板和视觉预处理器也必须支持工具结果图片；支持普通图片输入不代表支持工具图片。

针对 GLM-5.3-Flash 的默认模板，预处理会将调用 ID 明确、无歧义的工具结果分组按 assistant 的调用顺序排列，再进行模板渲染和图片收集。该排序规则仅用于 `Glm5NextForConditionalGeneration`，其他架构保留输入顺序；它不会自动推导任意或自定义 Jinja 模板的排序规则。InternVL3 和 InternVL3.5 会处理 user 和 tool 消息中的图片，并保留工具元数据；另一条消息中的旧式 `<IMAGE_TOKEN>` 占位符也不会影响当前消息的图片处理。

对应的工具结果格式见 [Messages API](../llm/api_server_anthropic.md) 和 [Responses API](../llm/api_server_responses.md) 文档。

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

> **注意：** 原生视频输入目前仅支持 **Qwen3-VL**、**Qwen3.5**、**Qwen3-Omni**、**InternS1-Pro** 和 **Intern-S2-Preview** 模型。

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

> **注意：** 原生视频输入目前仅支持 **Qwen3-VL**、**Qwen3.5**、**Qwen3-Omni**、**InternS1-Pro** 和 **Intern-S2-Preview** 模型。

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

> **注意：** 原生图像/视频混合输入目前仅支持 **Qwen3-VL**、**Qwen3.5**、**Qwen3-Omni**、**InternS1-Pro** 和 **Intern-S2-Preview** 模型。

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

## 单个音频

> **注意：** 音频输入目前仅支持 **Qwen3-Omni** 模型。

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
                'type': 'audio_url',
                'audio_url': {
                    'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav',
                },
            },
            {
                'type': 'text',
                'text': '描述这段音频。',
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

## 多个音频

> **注意：** 音频输入目前仅支持 **Qwen3-Omni** 模型。

<details>
<summary>完整示例</summary>

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
                'text': '比较这两段音频，有哪些相似点和不同点？',
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

当 API 服务面向不可信客户端时，建议在启动服务时使用 `--allowed-media-domains` 限制 HTTP(S) 媒体 URL 可访问的域名，降低 SSRF 风险。该参数按精确 hostname 匹配；例如 `--allowed-media-domains example.com` 只允许 `https://example.com/...`，不允许 `https://cdn.example.com/...`。

```shell
lmdeploy serve api_server <model_path> --allowed-media-domains example.com
```

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
<summary>Base64 编码示例（音频）</summary>

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
            {'type': 'text', 'text': '描述这段音频。'},
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

- **`mm_processor_kwargs`**：按 `image`、`video`、`audio` 模态传递模型对应的 HF processor 参数
- **`media_io_kwargs`**：控制媒体加载方式（如视频帧采样率/帧数和音频采样率）

两者均通过 `extra_body` 作为请求体中的额外字段传入 API，或在使用 pipeline API 时直接传给 `pipe()`。

`/generate`、`/v1/chat/completions` 和 Anthropic 兼容的 `/v1/messages` 均接受 JSON 请求体顶层的这两个字段；SDK 的 `extra_body` 会合并到请求体。

原生 GLM-5.3-Flash processor 使用 **token 预算**，而不是像素预算：

```json
{"mm_processor_kwargs": {"image": {"min_image_tokens": 64, "max_image_tokens": 64}}}
```

视频空间处理也可在 `video` 下使用相同的 token 预算参数。覆盖只影响当前请求，不修改共享 processor 默认值；可以只传一侧边界，另一侧使用默认值。边界必须为正整数，且最小值不能大于最大值。

原生 GLM 不使用通用 `size` 契约，因此 `min_pixels`、`max_pixels`、`size` 会被明确拒绝，不再静默忽略。Qwen 和旧版 GLM-4V 兼容 processor 保留下面的像素预算写法；兼容 processor 不支持原生 GLM token 预算覆盖。XTuner 的 GLM tokenizer 目前将 token 预算配置命名为 `min_pixels/max_pixels`，接入时应将其值映射到请求的 `min_image_tokens/max_image_tokens`，不能直接复制字段名。

### 视频抽帧与向后兼容

对于视频文件和 URL，LMDeploy 的媒体加载器在调用 HF processor **之前**解码并抽帧。使用 `media_io_kwargs.video.fps/num_frames` 控制抽帧。加载器默认最多取 32 帧；正值 `fps` 与 `num_frames` 同时起作用时，采用更严格的帧数限制。processor 自身的抽帧默认值不会自动配置媒体加载器。

为兼容旧请求，`mm_processor_kwargs.video` 中的 `fps`、`num_frames`、`min_frames`、`max_frames`、`do_sample_frames` 和 `video_metadata` 会被**忽略并警告**，而不是报错或自动移到 `media_io_kwargs`。HF 二次抽帧仍被关闭，元数据仍由媒体加载器提供。`min_frames/max_frames` 目前没有对应的加载器选项。

对于 Qwen、InternS2 等基于 `size` 的 processor，保留已有 `min_pixels/max_pixels` 语义：与 `size` 同时提供时，像素别名优先；最小值大于最大值时回退到 processor 默认值。其他受支持的 HF 选项在显式传入后会实际生效。上文原生 GLM 对像素字段的拒绝是单独的兼容性变化，不适用于基于 `size` 的 processor。

如果调用者已在外部抽帧并发送 `video/jpeg` 帧序列，LMDeploy 会保留收到的全部帧，不再套用普通视频加载器的帧数限制。该路径下 `num_frames` 不会裁减序列，`fps` 仅用于构造缺省元数据，不用于选帧。

这些规则描述的是服务侧行为，不代表训练侧与服务侧已完全一致。XTuner 与 LMDeploy 即使使用相同参数，也可能选出不同的帧或时间戳；还需要核对抽帧算法及元数据。

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

if __name__ == '__main__':
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
