import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.model import MODELS
from lmdeploy.vl.model.builder import load_vl_model


def get_model_and_chat_template(model_path, trust_remote_code=False):
    if os.getenv('LMDEPLOY_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download
    elif os.getenv('LMDEPLOY_USE_OPENMIND_HUB', 'False').lower() == 'true':
        from openmind_hub import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    model_path = snapshot_download(model_path, allow_patterns=['*.json', '*.py', '*.txt', '*.model', '*.jinja'])
    model = load_vl_model(model_path=model_path, with_llm=False, backend='pytorch',
                          trust_remote_code=trust_remote_code)
    chat_template = MODELS.module_dict['hf'](model_path=model_path, trust_remote_code=trust_remote_code)
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
        models = [get_model_and_chat_template(model_path, trust_remote_code=True) for model_path in model_list]
        return models

    def test_proc_messages(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=True)
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            # InternVL-HF and InternS1 models pad <img> and </img> internally
            reference = reference.replace('<IMG_CONTEXT>', '<img><IMG_CONTEXT></img>')
            prompt, _ = model.proc_messages(mock_messages, chat_template)
            assert prompt == reference

    def test_proc_pure_img_messages(self, models, mock_pure_img_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=True)
            reference = model.processor.apply_chat_template(mock_pure_img_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            # InternVL-HF and InternS1 models pad <img> and </img> internally
            reference = reference.replace('<IMG_CONTEXT>', '<img><IMG_CONTEXT></img>')
            prompt, _ = model.proc_messages(mock_pure_img_messages, chat_template)
            assert prompt == reference

    def test_proc_pure_text_messages(self, models, mock_pure_text_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=True)
            reference = model.processor.apply_chat_template(mock_pure_text_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_text_messages, chat_template)
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
        ]
        models = [get_model_and_chat_template(model_path) for model_path in model_list]
        return models

    def test_proc_messages(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=False)
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_messages, chat_template)
            assert prompt == reference

    def test_pure_img_messages(self, models, mock_pure_img_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=False)
            reference = model.processor.apply_chat_template(mock_pure_img_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_img_messages, chat_template)
            assert prompt == reference

    def test_pure_text_messages(self, models, mock_pure_text_messages):
        for model, chat_template in models:
            model.build_preprocessor(trust_remote_code=False)
            reference = model.processor.apply_chat_template(mock_pure_text_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt, _ = model.proc_messages(mock_pure_text_messages, chat_template)
            assert prompt == reference


class TestQwen3VLChatTemplate:

    @pytest.fixture(scope='module')
    def models(self):
        model_list = [
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

    def test_get_input_prompt(self, models, mock_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt = model.get_input_prompt(mock_messages, chat_template)
            assert prompt == reference

    def test_pure_img_messages(self, models, mock_pure_img_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_img_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt = model.get_input_prompt(mock_pure_img_messages, chat_template)
            assert prompt == reference

    def test_pure_text_messages(self, models, mock_pure_text_messages):
        for model, chat_template in models:
            model.build_preprocessor()
            reference = model.processor.apply_chat_template(mock_pure_text_messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False,
                                                            return_dict=True)
            prompt = model.get_input_prompt(mock_pure_text_messages, chat_template)
            assert prompt == reference


class TestGLM5NextProcessorCompatibility:

    @pytest.fixture
    def model(self):
        from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel

        model = GLM5NextVisionModel.__new__(GLM5NextVisionModel)
        model.model_path = 'glm5-next'
        model.hf_config = SimpleNamespace(image_token_id=42, video_token_id=43)
        return model

    @pytest.mark.parametrize('trust_remote_code', [False, True])
    @pytest.mark.parametrize('mode', ['native', 'missing_image', 'missing_video', ImportError, ValueError])
    def test_build_preprocessor(self, monkeypatch, model, trust_remote_code, mode):
        from lmdeploy.vl.model import glm5_next

        processor = SimpleNamespace(image_processor=object(), video_processor=object(),
                                    image_token='<image>', video_token='<video>')
        loader = Mock(return_value=processor)
        if isinstance(mode, type):
            loader.side_effect = mode('unsupported processor')
        elif mode.startswith('missing_'):
            loader.return_value = SimpleNamespace(**{
                key: value for key, value in vars(processor).items()
                if key != mode.removeprefix('missing_') + '_processor'
            })
        compat = Mock(return_value=processor)
        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', loader)
        monkeypatch.setattr(model, '_build_compat_processor', compat)

        model.build_preprocessor(trust_remote_code=trust_remote_code)

        loader.assert_called_once_with(model.model_path, trust_remote_code=trust_remote_code)
        if mode == 'native':
            compat.assert_not_called()
        else:
            compat.assert_called_once_with(trust_remote_code)
        assert model.processor is processor
        assert model.mm_tokens.image_token_id == 42
        assert model.mm_tokens.video_token_id == 43

    @pytest.mark.parametrize('error', [OSError, RuntimeError])
    def test_unrelated_load_error_propagates(self, monkeypatch, model, error):
        from lmdeploy.vl.model import glm5_next

        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', Mock(side_effect=error('load failed')))
        compat = Mock()
        monkeypatch.setattr(model, '_build_compat_processor', compat)
        with pytest.raises(error, match='load failed'):
            model.build_preprocessor()
        compat.assert_not_called()

    def test_compat_error_propagates(self, monkeypatch, model):
        from lmdeploy.vl.model import glm5_next

        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained',
                            Mock(side_effect=ValueError('unsupported processor')))
        compat = Mock(side_effect=ImportError('missing GLM-4V components'))
        monkeypatch.setattr(model, '_build_compat_processor', compat)
        with pytest.raises(ImportError, match='missing GLM-4V components'):
            model.build_preprocessor()
        compat.assert_called_once_with(False)
