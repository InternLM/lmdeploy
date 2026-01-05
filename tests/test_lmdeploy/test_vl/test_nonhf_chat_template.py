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


class TestInternVLChatTemplate:

    @pytest.fixture(scope='module')
    def internvl3_5(self):
        model_list = [
            'OpenGVLab/InternVL3_5-241B-A28B',
            'OpenGVLab/InternVL3_5-30B-A3B',
            'OpenGVLab/InternVL3_5-38B',
            'OpenGVLab/InternVL3_5-14B',
            'OpenGVLab/InternVL3_5-8B',
            'OpenGVLab/InternVL3_5-4B',
            'OpenGVLab/InternVL3_5-2B',
            'OpenGVLab/InternVL3_5-1B',
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    @pytest.fixture(scope='module')
    def internvl3(self):
        model_list = [
            'OpenGVLab/InternVL3-78B',
            'OpenGVLab/InternVL3-38B',
            'OpenGVLab/InternVL3-14B',
            'OpenGVLab/InternVL3-8B',
            # "OpenGVLab/InternVL3-9B",  # <s>
            'OpenGVLab/InternVL3-2B',
            'OpenGVLab/InternVL3-1B',
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    @pytest.fixture(scope='module')
    def internvl2_5(self):
        model_list = [
            'OpenGVLab/InternVL2_5-78B',
            'OpenGVLab/InternVL2_5-38B',
            # "OpenGVLab/InternVL2_5-26B",  # <s>
            # "OpenGVLab/InternVL2_5-8B",  # <s>
            'OpenGVLab/InternVL2_5-4B',
            # "OpenGVLab/InternVL2_5-2B",  # <s>
            'OpenGVLab/InternVL2_5-1B',
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    @pytest.fixture(scope='module')
    def internvl2(self):
        model_list = [
            'OpenGVLab/InternVL2-Llama3-76B',
            'OpenGVLab/InternVL2-40B',
            'OpenGVLab/InternVL2-26B',
            'OpenGVLab/InternVL2-8B',
            # "OpenGVLab/InternVL2-4B",  # <|user|> not <|im_start|>
            'OpenGVLab/InternVL2-2B',
            'OpenGVLab/InternVL2-1B',
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

    @pytest.fixture(scope='module')
    def mock_IMAGE_TOKEN_messages(self):
        return [
            dict(role='system', content='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'),
            dict(role='user',
                 content=[
                     dict(type='text', text='<IMAGE_TOKEN>\nDescribe the following images in detail'),
                     dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg'))
                 ]),
        ]

    def test_internvl3_5(self, internvl3_5, mock_messages):
        reference = """<|im_start|>user
Describe the following images in detail<img><IMG_CONTEXT></img>
<img><IMG_CONTEXT></img>
How many cats are there in total?<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl3_5:
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)

            assert prompt == reference

    def test_internvl3_5_backward_compatibility(self, internvl3_5, mock_IMAGE_TOKEN_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
<img><IMG_CONTEXT></img>
Describe the following images in detail<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl3_5:
            prompt, _ = model.proc_messages(mock_IMAGE_TOKEN_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_internvl3(self, internvl3, mock_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
Describe the following images in detail<img><IMG_CONTEXT></img>
<img><IMG_CONTEXT></img>
How many cats are there in total?<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl3:
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_internvl3_backward_compatibility(self, internvl3, mock_IMAGE_TOKEN_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
<img><IMG_CONTEXT></img>
Describe the following images in detail<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl3:
            prompt, _ = model.proc_messages(mock_IMAGE_TOKEN_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_internvl2_5(self, internvl2_5, mock_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
Describe the following images in detail<img><IMG_CONTEXT></img>
<img><IMG_CONTEXT></img>
How many cats are there in total?<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl2_5:
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_internvl2_5_backward_compatibility(self, internvl2_5, mock_IMAGE_TOKEN_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
<img><IMG_CONTEXT></img>
Describe the following images in detail<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl2_5:
            prompt, _ = model.proc_messages(mock_IMAGE_TOKEN_messages, chat_template, sequence_start=True)
            assert prompt == reference

    def test_internvl2(self, internvl2, mock_messages):
        reference = """<|im_start|>user
Describe the following images in detail<img><IMG_CONTEXT></img>
<img><IMG_CONTEXT></img>
How many cats are there in total?<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl2:
            # Let sequence_start=False to avoid the begin-of-prompt token, such as <|begin_of_text|>, <s>
            prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=False)
            assert prompt == reference

    def test_internvl2_backward_compatibility(self, internvl2, mock_IMAGE_TOKEN_messages):
        reference = """<|im_start|>system
你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>
<|im_start|>user
<img><IMG_CONTEXT></img>
Describe the following images in detail<|im_end|>
<|im_start|>assistant
"""
        for model, chat_template in internvl2:
            # Let sequence_start=False to avoid the begin-of-prompt token, such as <|begin_of_text|>, <s>
            prompt, _ = model.proc_messages(mock_IMAGE_TOKEN_messages, chat_template, sequence_start=False)
            assert prompt == reference
