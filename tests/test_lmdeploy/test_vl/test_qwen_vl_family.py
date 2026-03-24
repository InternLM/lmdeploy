# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.archs import get_task
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.vl.model.qwen3 import Qwen3VLModel, resolve_qwen_vl_family_automodel


@pytest.mark.parametrize('arch,expected_block', [
    ('Qwen3VLForConditionalGeneration', 'Qwen3VLVisionBlock'),
    ('Qwen3VLMoeForConditionalGeneration', 'Qwen3VLMoeVisionBlock'),
    ('Qwen3_5ForConditionalGeneration', 'Qwen3_5VisionBlock'),
    ('Qwen3_5MoeForConditionalGeneration', 'Qwen3_5MoeVisionBlock'),
])
def test_resolve_qwen_vl_family_automodel(arch, expected_block):
    cls, no_split = resolve_qwen_vl_family_automodel(arch)
    assert cls is not None
    assert expected_block in no_split


def test_resolve_unknown_arch_raises():
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_qwen_vl_family_automodel('NotAModel')


def test_get_task_routes_qwen3_vl_to_vl_engine(monkeypatch):
    cfg = SimpleNamespace(to_dict=lambda: {'architectures': ['Qwen3VLForConditionalGeneration']})
    monkeypatch.setattr('lmdeploy.archs.get_model_arch', lambda _path: ('Qwen3VLForConditionalGeneration', cfg))

    task, pipeline_class = get_task('/fake-model', TurbomindEngineConfig())
    assert task == 'vlm'
    assert pipeline_class.__name__ == 'VLAsyncEngine'


class _DummyChatTemplate:

    def __init__(self, prompt):
        self.prompt = prompt

    def messages2prompt(self, messages, sequence_start, **kwargs):
        return self.prompt


class _DummyTokenizer:

    def encode(self, text, add_bos=False):
        tokens = [] if not text else [len(text)]
        if add_bos:
            return [0] + tokens
        return tokens


def _build_qwen3_vl_stub():
    model = Qwen3VLModel.__new__(Qwen3VLModel)
    model.image_token = '<|image_pad|>'
    model.image_token_id = 151655
    model.contains_video_input = False
    return model


def test_qwen3_vl_to_turbomind_uses_image_token_placeholder():
    model = _build_qwen3_vl_stub()
    tokenizer = _DummyTokenizer()
    prompt = 'prefix<|vision_start|><|image_pad|><|vision_end|>suffix'
    chat_template = _DummyChatTemplate(prompt)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    image_embed = torch.randn(1, 4)
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'image',
            'data': object(),
        }],
    }, {
        'role': 'preprocess',
        'content': [{
            'image_grid_thw': image_grid_thw,
        }],
    }, {
        'role': 'forward',
        'content': [image_embed],
    }]

    info = model.to_turbomind(messages, chat_template, tokenizer, sequence_start=True)

    begin = len(tokenizer.encode('prefix<|vision_start|>', add_bos=True))
    assert info['input_embedding_ranges'] == [[begin, begin + image_embed.shape[0]]]
    assert len(info['input_embeddings']) == 1
    assert info['input_meta']['mrope_position_ids'].shape[1] == len(info['input_ids'])


def test_qwen3_vl_to_turbomind_rejects_video():
    model = _build_qwen3_vl_stub()
    model.contains_video_input = True
    messages = [{
        'role': 'preprocess',
        'content': [{
            'video_grid_thw': torch.tensor([[1, 2, 2]]),
        }],
    }]

    with pytest.raises(NotImplementedError, match='supports images only'):
        model.to_turbomind(messages, _DummyChatTemplate(''), _DummyTokenizer(), sequence_start=True)
