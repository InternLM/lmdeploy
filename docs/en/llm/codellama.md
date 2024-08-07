# codellama

## Introduction

[codellama](https://github.com/facebookresearch/codellama) features enhanced coding capabilities. It can generate code and natural language about code, from both code and natural language prompts (e.g., “Write me a function that outputs the fibonacci sequence”). It can also be used for code completion and debugging. It supports many of the most popular programming languages used today, including Python, C++, Java, PHP, Typescript (Javascript), C#, Bash and more.

There are three sizes (7b, 13b, 34b) as well as three flavours (base model, Python fine-tuned, and instruction tuned) released on [HuggingFace](https://huggingface.co/codellama).

| Base Model                                                                      | Python                                                                                        | Instruct                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)   | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)   | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)   |
| [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

The correspondence between the model and capabilities is:

| models     | code completion | infilling         | instructions / chat | python specialist |
| ---------- | --------------- | ----------------- | ------------------- | ----------------- |
| Base Model | Y               | Y(7B,13B), N(34B) | N                   | N                 |
| Python     | Y               | N                 | N                   | Y                 |
| Instruct   | Y               | Y(7B,13B), N(34B) | Y                   | N                 |

## Inference

Based on the above table, this section shows how to utilize CodeLlama's capabilities by examples

### Completion

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

### Infilling

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

### Chat

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

### Python specialist

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

## Quantization

TBD

## Serving

Prepare a chat template json file, for instance "codellama.json", with the following content:

```json
{
    "model_name": "codellama",
    "capability": "completion"
}
```

Then launch the service as follows:

```shell
lmdeploy serve api_server meta-llama/CodeLlama-7b-Instruct-hf --chat-template codellama.json
```

After the service is launched successfully, you can access the service with `openai` package:

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

Regarding the detailed information of the api_server, you can refer to the [guide](../llm/api_server.md).
