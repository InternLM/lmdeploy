# 结构化输出

目前只有 Pytorch 后端具有该能力。所以无论是使用 pipline 还是使用 api_server，请指定使用 pytorch 后端。

## pipeline

```python
from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

model = 'internlm/internlm2-chat-1_8b'
guide = {
    'type': 'object',
    'properties': {
        'name': {
            'type': 'string'
        },
        'skills': {
            'type': 'array',
            'items': {
                'type': 'string',
                'maxLength': 10
            },
            'minItems': 3
        },
        'work history': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'company': {
                        'type': 'string'
                    },
                    'duration': {
                        'type': 'string'
                    }
                },
                'required': ['company']
            }
        }
    },
    'required': ['name', 'skills', 'work history']
}
pipe = pipeline(model, backend_config=PytorchEngineConfig(), log_level='INFO')
gen_config = GenerationConfig(
    response_format=dict(type='json_schema', json_schema=dict(name='test', schema=guide)))
response = pipe(['Make a self introduction please.'], gen_config=gen_config)
print(response)
```

## api_server

首先，先启动 InternLM2 模型的 api_server 服务。

```shell
lmdeploy serve api_server internlm/internlm2-chat-1_8b --backend pytorch
```

客户端可以使用 OpenAI 的 python 包进行测试：

```python
from openai import OpenAI
guide = {
    'type': 'object',
    'properties': {
        'name': {
            'type': 'string'
        },
        'skills': {
            'type': 'array',
            'items': {
                'type': 'string',
                'maxLength': 10
            },
            'minItems': 3
        },
        'work history': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'company': {
                        'type': 'string'
                    },
                    'duration': {
                        'type': 'string'
                    }
                },
                'required': ['company']
            }
        }
    },
    'required': ['name', 'skills', 'work history']
}
response_format=dict(type='json_schema',  json_schema=dict(name='test',schema=guide))
messages = [{'role': 'user', 'content': 'Make a self-introduction please.'}]
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    response_format=response_format,
    top_p=0.8)
print(response)
```

输出结果是一个 json 格式的回答。
