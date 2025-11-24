import os

import pytest

from lmdeploy.model import MODELS, best_match_model
from lmdeploy.vl.model.builder import load_vl_model


def get_model_and_chat_template(model_path):
    if os.getenv('LMDEPLOY_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download
    elif os.getenv('LMDEPLOY_USE_OPENMIND_HUB', 'False').lower() == 'true':
        from openmind_hub import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    model_path = snapshot_download(model_path, allow_patterns=['*.json', '*.py', '*.txt', '*.model', '*.jinja'])
    model = load_vl_model(model_path=model_path, with_llm=False, backend='pytorch')
    chat_template_name = best_match_model(model_path)
    chat_template = MODELS.module_dict[chat_template_name](model_path=model_path)
    return model, chat_template


class TestVLHFChatTemplate:

    @pytest.fixture(scope='module')
    def models(self):
        model_list = [
            'OpenGVLab/InternVL3_5-1B-HF',
            'OpenGVLab/InternVL3_5-2B-HF',
            'OpenGVLab/InternVL3_5-4B-HF',
            'OpenGVLab/InternVL3_5-8B-HF',
            'OpenGVLab/InternVL3_5-14B-HF',
            'OpenGVLab/InternVL3_5-38B-HF',
            'OpenGVLab/InternVL3_5-30B-A3B-HF',
            'OpenGVLab/InternVL3_5-241B-A28B-HF',
            'internlm/Intern-S1-mini',
            'internlm/Intern-S1',
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2-VL-72B-Instruct',
            'Qwen/Qwen2.5-VL-3B-Instruct',
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-32B-Instruct',
            'Qwen/Qwen2.5-VL-72B-Instruct',
            'Qwen/Qwen3-VL-2B-Instruct',
            'Qwen/Qwen3-VL-2B-Thinking',
            'Qwen/Qwen3-VL-4B-Instruct',
            'Qwen/Qwen3-VL-4B-Thinking',
            'Qwen/Qwen3-VL-8B-Instruct',
            'Qwen/Qwen3-VL-8B-Thinking',
            'Qwen/Qwen3-VL-32B-Instruct',
            'Qwen/Qwen3-VL-32B-Thinking',
            'Qwen/Qwen3-VL-30B-A3B-Instruct',
            'Qwen/Qwen3-VL-30B-A3B-Thinking',
            'Qwen/Qwen3-VL-235B-A22B-Instruct',
            'Qwen/Qwen3-VL-235B-A22B-Thinking',
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    @pytest.fixture(scope='module')
    def mock_messages(self):
        return [
            dict(role='user',
                 content=[
                     dict(type='text', text='Describe the following images in detail'),
                     dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                     dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                     dict(type='text', text='How many cats are there in total?')
                 ]),
        ]

    def test_proc_messages(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_none_text(self, models):
        for model, chat_template in models:
            model.build_preprocessor()
            messages = [
                dict(role='user',
                     content=[
                         dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                     ]),
            ]
            reference = model.processor.apply_chat_template(messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(messages, chat_template, sequence_start=True)
            assert prompt == reference
