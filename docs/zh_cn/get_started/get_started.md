# 快速开始

LMDeploy提供了快速安装、模型量化、离线批处理、在线推理服务等功能。每个功能只需简单的几行代码或者命令就可以完成。

本教程将展示 LMDeploy 在以下几方面的使用方法：

- LLM 模型和 VLM 模型的离线推理
- 搭建与 OpenAI 接口兼容的 LLM 或 VLM 模型服务
- 通过控制台命令行与 LLM 模型进行交互式聊天

在继续阅读之前，请确保你已经按照[安装指南](installation.md)安装了 lmdeploy。

## 离线批处理

### LLM 推理

```python
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

在构造 `pipeline` 时，如果没有指定使用 TurboMind 引擎或 PyTorch 引擎进行推理，LMDeploy 将根据[它们各自的能力](../supported_models/supported_models.md)自动分配一个，默认优先使用 TurboMind 引擎。

然而，你可以选择手动选择一个引擎。例如，

```python
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=TurbomindEngineConfig(
                    max_batch_size=32,
                    enable_prefix_caching=True,
                    cache_max_entry_count=0.8,
                    session_len=8192,
                ))
```

或者，

```python
from lmdeploy import pipeline, PytorchEngineConfig
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=PytorchEngineConfig(
                    max_batch_size=32,
                    enable_prefix_caching=True,
                    cache_max_entry_count=0.8,
                    session_len=8192,
                ))
```

```{note}
参数 "cache_max_entry_count" 显著影响 GPU 内存占用。它表示加载模型权重后 K/V 缓存占用的空闲 GPU 内存的比例。
默认值是 0.8。K/V 缓存分配方式是一次性申请，重复性使用，这就是为什么 pipeline 以及下文中的 api_server 在启动后会消耗大量 GPU 内存。
如果你遇到内存不足(OOM)错误的错误，可能需要考虑降低 cache_max_entry_count 的值。
```

当使用 `pipe()` 生成提示词的 token 时，你可以通过 `GenerationConfig` 设置采样参数，如下所示：

```python
from lmdeploy import GenerationConfig, pipeline

pipe = pipeline('internlm/internlm2_5-7b-chat')
prompts = ['Hi, pls intro yourself', 'Shanghai is']
response = pipe(prompts,
                gen_config=GenerationConfig(
                    max_new_tokens=1024,
                    top_p=0.8,
                    top_k=40,
                    temperature=0.6
                ))
```

在 `GenerationConfig` 中，`top_k=1` 或 `temperature=0.0` 表示贪心搜索。

有关 pipeline 的更多信息，请参考[这里](../llm/pipeline.md)

### VLM 推理

VLM 推理 pipeline 与 LLM 类似，但增加了使用 pipeline 处理图像数据的能力。例如，你可以使用以下代码片段对 InternVL 模型进行推理：

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL2-8B')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

在 VLM pipeline 中，默认的图像处理批量大小是 1。这可以通过 `VisionConfig` 调整。例如，你可以这样设置：

```python
from lmdeploy import pipeline, VisionConfig
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL2-8B',
                vision_config=VisionConfig(
                    max_batch_size=8
                ))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

然而，图像批量大小越大，OOM 错误的风险越大，因为 VLM 模型中的 LLM 部分会提前预分配大量的内存。

VLM pipeline 对于推理引擎的选择方式与 LLM pipeline 类似。你可以参考 [LLM 推理](#llm-推理)并结合两个引擎支持的 VLM 模型列表，手动选择和配置推理引擎。

## 模型服务

类似前文[离线批量推理](#离线批处理)，我们在本章节介绍 LLM 和 VLM 各自构建服务方法。

### LLM 模型服务

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

此命令将在本地主机上的端口 `23333` 启动一个与 OpenAI 接口兼容的模型推理服务。你可以使用 `--server-port` 选项指定不同的服务器端口。
更多选项，请通过运行 `lmdeploy serve api_server --help` 查阅帮助文档。这些选项大多与引擎配置一致。

要访问服务，你可以使用官方的 OpenAI Python 包 `pip install openai`。以下是演示如何使用入口点 v1/chat/completions 的示例：

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": " provide three suggestions about time management"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

我们鼓励你参考详细指南，了解关于[使用 Docker 部署服务](../llm/api_server.md)、[工具调用](../llm/api_server_tools.md)和其他更多功能的信息。

### VLM 模型服务

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B
```

```{note}
LMDeploy 复用了上游 VLM 仓库的视觉组件。而每个上游的 VLM 模型，它们的视觉模型可能互不相同，依赖库也各有区别。
因此，LMDeploy 决定不在自身的依赖列表中加入上游 VLM 库的依赖。如果你在使用 LMDeploy 推理 VLM 模型时出现 "ImportError" 的问题，请自行安装相关的依赖。
```

服务成功启动后，你可以以类似访问 `gptv4` 服务的方式访问 VLM 服务：

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', # A dummy api_key is required
                base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## 使用命令行与 LLM 模型对话

LMDeploy 提供了一个非常方便的 CLI 工具，供用户与 LLM 模型进行本地聊天。例如：

```shell
lmdeploy chat internlm/internlm2_5-7b-chat --backend turbomind
```

它的设计目的是帮助用户检查和验证 LMDeploy 是否支持提供的模型，聊天模板是否被正确应用，以及推理结果是否正确。

另外，`lmdeploy check_env` 收集基本的环境信息。在给 LMDeploy 提交问题报告时，这非常重要，因为它有助于我们更有效地诊断和解决问题。

如果你对它们的使用方法有任何疑问，你可以尝试使用 `--help` 选项获取详细信息。
