import os

import pytest

from lmdeploy.model import MODELS
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
    chat_template = MODELS.module_dict['hf'](model_path=model_path)
    return model, chat_template


@pytest.fixture(scope='module')
def mock_messages():
    return [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the following images in detail'),
                 dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                 dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                 dict(type='text', text='How many cats are there in total?')
             ]),
    ]


@pytest.fixture(scope='module')
def mock_pure_img_messages():
    return [
        dict(role='user',
             content=[
                 dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
             ]),
    ]


@pytest.fixture(scope='module')
def mock_pure_text_messages():
    return [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the following images in detail'),
                 dict(type='text', text='How many cats are there in total?'),
             ]),
    ]


class TestInternVLHFChatTemplate:

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
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    def test_proc_messages(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            # InternVL-HF and InternS1 models pad <img> and </img> internally
            reference = reference.replace('<IMG_CONTEXT>', '<img><IMG_CONTEXT></img>')
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_proc_pure_img_messages(self, models, mock_pure_img_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_img_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            # InternVL-HF and InternS1 models pad <img> and </img> internally
            reference = reference.replace('<IMG_CONTEXT>', '<img><IMG_CONTEXT></img>')
            prompt, _ = model.proc_messages(mock_pure_img_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_proc_pure_text_messages(self, models, mock_pure_text_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_text_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_text_messages, chat_template, sequence_start=True)
            assert prompt == reference


class TestQwenVLChatTemplate:

    @pytest.fixture(scope='module')
    def models(self):
        model_list = [
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

    def test_proc_messages(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_pure_img_messages(self, models, mock_pure_img_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_img_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_img_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_pure_text_messages(self, models, mock_pure_text_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_text_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_text_messages, chat_template, sequence_start=True)
            assert prompt == reference
