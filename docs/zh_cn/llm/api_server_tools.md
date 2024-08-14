# Tools

LMDeploy 支持 InternLM2, InternLM2.5 和 Llama3.1 模型的工具调用。

## 单轮调用

启动好模型的服务后，运行下面 demo 即可。

```python
from openai import OpenAI

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

client = OpenAI(api_key='YOUR_API_KEY',base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
```

## 多轮调用

### InternLM

一个完整的工具链调用过程可以通过下面的例子展示。

```python
from openai import OpenAI


def add(a: int, b: int):
    return a + b


def mul(a: int, b: int):
    return a * b


tools = [{
    'type': 'function',
    'function': {
        'name': 'add',
        'description': 'Compute the sum of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}, {
    'type': 'function',
    'function': {
        'name': 'mul',
        'description': 'Calculate the product of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}]
messages = [{'role': 'user', 'content': 'Compute (3+5)*2'}]

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func1_name = response.choices[0].message.tool_calls[0].function.name
func1_args = response.choices[0].message.tool_calls[0].function.arguments
func1_out = eval(f'{func1_name}(**{func1_args})')
print(func1_out)

messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content
})
messages.append({
    'role': 'environment',
    'content': f'3+5={func1_out}',
    'name': 'plugin'
})
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func2_name = response.choices[0].message.tool_calls[0].function.name
func2_args = response.choices[0].message.tool_calls[0].function.arguments
func2_out = eval(f'{func2_name}(**{func2_args})')
print(func2_out)
```

实际使用 InternLM2-Chat-7B 模型执行上述例子，可以得到下面的结果：

```
ChatCompletion(id='1', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='', role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='1', function=Function(arguments={'a': 3, 'b': 5}, name='add'), type='function')]))], created=1719369986, model='/nvme/shared_data/InternLM/internlm2-chat-7b', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=25, prompt_tokens=263, total_tokens=288))
8
ChatCompletion(id='2', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='', role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='2', function=Function(arguments={'a': 8, 'b': 2}, name='mul'), type='function')]))], created=1719369987, model='/nvme/shared_data/InternLM/internlm2-chat-7b', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=25, prompt_tokens=282, total_tokens=307))
16
```

### Llama3.1

Meta 在 [Llama3 的官方用户指南](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)中宣布（注：下文为原文的中文翻译）：

```{text}
有三个内置工具（brave_search、wolfram_alpha 和 code interpreter）可以使用系统提示词打开：

1. Brave Search：执行网络搜索的工具调用。
2. Wolfram Alpha：执行复杂数学计算的工具调用。
3. Code Interpreter：使模型能够输出 Python 代码的功能。
```

此外，它还警告说：“注意： 我们建议使用 Llama 70B-instruct 或 Llama 405B-instruct 用于结合对话和工具调用的应用。Llama 8B-Instruct 无法可靠地在工具调用定义的同时维持对话。它可以用于零样本工具调用，但在模型和用户之间的常规对话中，应移除工具指令。”（注：引号中内容为原文的中文翻译）

因此，我们使用 [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) 来展示如何通过 LMDeploy的`api_server`调用模型的工具能力.

在 A100-SXM-80G 节点上，可以按照以下方式启动服务：

```shell
lmdeploy serve api_server /the/path/of/Meta-Llama-3.1-70B-Instruct/model --tp 4
```

有关 api_server 的详细介绍，请参考[此处](./api_server.md)的详细文档。

以下代码示例展示了如何使用 "Wolfram Alpha" 工具。假设你已经在[Wolfram Alpha](https://www.wolframalpha.com) 网站上注册并获取了 API 密钥。请确保拥有一个有效的 API 密钥，以便访问 Wolfram Alpha 提供的服务。

```python
from openai import OpenAI
import requests


def request_llama3_1_service(messages):
    client = OpenAI(api_key='YOUR_API_KEY',
                    base_url='http://0.0.0.0:23333/v1')
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8,
        top_p=0.8,
        stream=False)
    return response.choices[0].message.content


# The role of "system" MUST be specified, including the required tools
messages = [
    {
        "role": "system",
        "content": "Environment: ipython\nTools: wolfram_alpha\n\n Cutting Knowledge Date: December 2023\nToday Date: 23 Jul 2024\n\nYou are a helpful Assistant." # noqa
    },
    {
        "role": "user",
        "content": "Can you help me solve this equation: x^3 - 4x^2 + 6x - 24 = 0"  # noqa
    }
]

# send request to the api_server of llama3.1-70b and get the response
# the "assistant_response" is supposed to be:
# <|python_tag|>wolfram_alpha.call(query="solve x^3 - 4x^2 + 6x - 24 = 0")
assistant_response = request_llama3_1_service(messages)
print(assistant_response)

# Call the API of Wolfram Alpha with the query generated by the model
app_id = 'YOUR-Wolfram-Alpha-API-KEY'
params = {
    "input": assistant_response,
    "appid": app_id,
    "format": "plaintext",
    "output": "json",
}

wolframalpha_response = requests.get(
    "https://api.wolframalpha.com/v2/query",
    params=params
)
wolframalpha_response = wolframalpha_response.json()

# Append the contents obtained by the model and the wolframalpha's API
# to "messages", and send it again to the api_server
messages += [
    {
        "role": "assistant",
        "content": assistant_response
    },
    {
        "role": "ipython",
        "content": wolframalpha_response
    }
]

assistant_response = request_llama3_1_service(messages)
print(assistant_response)
```
