# Code Llama

## 模型介绍

[codellama](https://github.com/facebookresearch/codellama) 支持很多种编程语言，包括 Python, C++, Java, PHP, Typescript (Javascript), C#, Bash 等等。具备代码续写、代码填空、对话、python专项等 4 种能力。

它在 [HuggingFace](https://huggingface.co/codellama) 上发布了基座模型，Python模型和指令微调模型：

| 基座模型                                                                        | Python微调模型                                                                                | 指令模型                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)   | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)   | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)   |
| [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

模型和能力的对应关系为：

| 模型           | 代码续写 | 代码填空          | 对话 | Python专项 |
| -------------- | -------- | ----------------- | ---- | ---------- |
| 基座模型       | Y        | Y(7B,13B), N(34B) | N    | N          |
| Python微调模型 | Y        | N                 | N    | Y          |
| 指令微调模型   | Y        | Y(7B,13B), N(34B) | Y    | N          |

## 推理

根据前文模型的能力表，在本小节中，我们讲通过具体的示例展示使用 CodeLlama 各能力的方法

### 代码续写

```python
from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig

pipe = pipeline('meta-llama/CodeLlama-7b-hf',
                chat_template_config=ChatTemplateConfig(
                    model_name='codellama',
                    capability='completion'
                ))

response = pipe(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    gen_config=GenerationConfig(
        top_k=10,
        temperature=0.1,
        top_p=0.95
    )
)
print(response.text)
```

### 代码填空

```python
from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig

pipe = pipeline('meta-llama/CodeLlama-7b-hf',
                chat_template_config=ChatTemplateConfig(
                    model_name='codellama',
                    capability='infilling'
                ))

prompt = """
def remove_non_ascii(s: str) -> str:
    \"\"\"
    <FILL>
    \"\"\"
    return result
"""
response = pipe(
    prompt,
    gen_config=GenerationConfig(
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=500
    )
)
print(response.text)
```

### 对话

```python
from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig

pipe = pipeline('meta-llama/CodeLlama-7b-Instruct-hf',
                chat_template_config=ChatTemplateConfig(
                    model_name='codellama',
                    capability='chat'
                ))

response = pipe(
    'implement quick sort in C++',
    gen_config=GenerationConfig(
        top_k=10,
        temperature=0.1,
        top_p=0.95
    )
)
print(response.text)
```

### Python 专项

```python
from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig

pipe = pipeline('meta-llama/CodeLlama-7b-Python-hf',
                chat_template_config=ChatTemplateConfig(
                    model_name='codellama',
                    capability='python'
                ))

response = pipe(
    'implement quick sort',
    gen_config=GenerationConfig(
        top_k=10,
        temperature=0.1,
        top_p=0.95
    )
)
print(response.text)
```

## 量化

TBD

## 服务

准备好对话模板文件，比如说“codellama.json”，参考如下示例，填写 CodeLlama 的能力：

```json
{
    "model_name": "codellama",
    "capability": "completion"
}
```

然后，启动推理服务：

```shell
lmdeploy serve api_server meta-llama/CodeLlama-7b-Instruct-hf --chat-template codellama.json
```

在服务启动成功后，可以通过`openai`客户端接口，访问服务：

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
    {"role": "user", "content": "import socket\n\ndef ping_exponential_backoff(host: str):"},
  ],
    temperature=0.1,
    top_p=0.95,
    max_tokens=500
)
print(response)
```

关于 api_server 的详细介绍，请参考[这份](../llm/api_server.md)文档。
