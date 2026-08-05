import asyncio

import pytest

from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.model.base import VisionModel
from lmdeploy.vl.model.mllama import MllamaVLModel
from lmdeploy.vl.model.qwen import QwenVisionModel
from lmdeploy.vl.model.qwen2 import Qwen2VLModel
from lmdeploy.vl.model.xcomposer2 import ModelType, Xcomposer2VisionModel

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string'
                }
            },
        },
    },
}]
EXPECTED_PROMPT = 'tools=get_weather;enable_thinking=False'


class _ToolAwareChatTemplate:

    @staticmethod
    def messages2prompt(messages, sequence_start, tools=None, **kwargs):
        tool_names = ','.join(tool['function']['name'] for tool in tools or [])
        return f'tools={tool_names};enable_thinking={kwargs.get("enable_thinking")}'


class _NewPreprocessModel:
    get_input_prompt = staticmethod(VisionModel.get_input_prompt)


class _NewPreprocessEncoder:
    _uses_new_preprocess = True
    model = _NewPreprocessModel()

    @staticmethod
    async def preprocess(messages, input_prompt=None, mm_processor_kwargs=None):
        return {'prompt': input_prompt}


class _LegacyQwen2Encoder:
    _uses_new_preprocess = False
    wrap_for_pytorch = ImageEncoder.wrap_for_pytorch

    def __init__(self):
        self.model = object.__new__(Qwen2VLModel)
        self.model.image_token = '<|image_pad|>'
        self.model.to_pytorch_aux = lambda messages, prompt, image_token, tokenizer, sequence_start: {
            'prompt': prompt
        }

    @staticmethod
    async def preprocess(messages, input_prompt=None, mm_processor_kwargs=None):
        return messages


def _messages():
    return [{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': 'What is the weather in this city?'
            },
            {
                'type': 'image_data',
                'image_data': {
                    'data': object()
                }
            },
        ],
    }]


def _parsed_messages():
    return [{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': 'What is the weather in this city?'
            },
            {'type': 'image'},
        ],
    }]


def _get_prompt(processor):
    result = asyncio.run(
        processor.get_prompt_input(prompt=_messages(),
                                   do_preprocess=True,
                                   sequence_start=True,
                                   adapter_name='',
                                   tools=TOOLS,
                                   chat_template_kwargs={'enable_thinking': False}))
    return result['prompt']


@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'])
def test_tools_reach_new_multimodal_preprocess(backend):
    processor = MultimodalProcessor(tokenizer=None,
                                    chat_template=_ToolAwareChatTemplate(),
                                    vl_encoder=_NewPreprocessEncoder(),
                                    backend=backend)

    assert _get_prompt(processor) == EXPECTED_PROMPT


def test_tools_reach_legacy_multimodal_preprocess():
    processor = MultimodalProcessor(tokenizer=None,
                                    chat_template=_ToolAwareChatTemplate(),
                                    vl_encoder=_LegacyQwen2Encoder(),
                                    backend='pytorch')

    assert _get_prompt(processor) == EXPECTED_PROMPT


@pytest.mark.parametrize(
    ('model_cls', 'model_attrs'),
    [
        pytest.param(MllamaVLModel, {}, id='mllama'),
        pytest.param(QwenVisionModel, {}, id='qwen'),
        pytest.param(Xcomposer2VisionModel, {'model_type': ModelType.XCOMPOSER2}, id='xcomposer2'),
    ],
)
def test_tools_reach_v014_multimodal_templates(model_cls, model_attrs):
    model = object.__new__(model_cls)
    for name, value in model_attrs.items():
        setattr(model, name, value)
    model.to_pytorch_aux = lambda messages, prompt, image_token, tokenizer, sequence_start: {'prompt': prompt}

    result = model.to_pytorch(_parsed_messages(),
                              _ToolAwareChatTemplate(),
                              tokenizer=None,
                              sequence_start=True,
                              tools=TOOLS,
                              chat_template_kwargs={'enable_thinking': False})

    assert result['prompt'] == EXPECTED_PROMPT
