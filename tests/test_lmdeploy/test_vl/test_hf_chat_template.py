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
        model.backend = ''
        model.hf_config = SimpleNamespace(image_token_id=42, video_token_id=43)
        return model

    @pytest.fixture
    def real_model(self, model, monkeypatch):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
        from transformers.models.glm5_next.processing_glm5_next import Glm5NextProcessor
        from transformers.models.glm5_next.video_processing_glm5_next import Glm5NextVideoProcessor

        from lmdeploy.vl.model import glm5_next

        special = ['<|image|>', '<|video|>', '<|begin_of_image|>', '<|end_of_image|>',
                   '<|begin_of_video|>', '<|end_of_video|>']
        vocab = {f'word_{i}': i for i in range(42)}
        vocab.update({token: i for i, token in enumerate(special, 42)})
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='word_0')),
                                           unk_token='word_0', pad_token='word_0', additional_special_tokens=special)
        processor = Glm5NextProcessor(image_processor=Glm5NextImageProcessor(),
                                      video_processor=Glm5NextVideoProcessor(), tokenizer=tokenizer)
        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', lambda *a, **k: processor)
        model.build_preprocessor()
        assert model._shared_video_token
        return model

    @pytest.mark.parametrize('raw', [False, True])
    def test_cached_images_match_native_processor(self, real_model, raw):
        import torch
        from PIL import Image

        model = real_model
        model.backend = 'pytorch'
        model.mm_feature_dtype = torch.bfloat16
        model.image_cache.max_bytes = 16 * 1024**2
        images = [Image.new('RGB', (56, 112), 'red'), Image.new('RGB', (112, 56), 'blue')]
        messages = [dict(role='user', content=[dict(type='image', data=image) for image in images])]
        prompt = '<|begin_of_image|><|image|><|end_of_image|>word_3<|begin_of_image|><|image|><|end_of_image|>'
        input_prompt = model.processor.tokenizer.encode(prompt, add_special_tokens=False) if raw else prompt
        options = {'min_image_tokens': 32, 'max_image_tokens': 64}
        expected = model.processor(text=[prompt], images=images, images_kwargs=options, return_tensors='pt')
        cold = model.preprocess(messages, input_prompt, {'image': options})
        warm = model.preprocess(messages, input_prompt, {'image': options})
        for result in (cold, warm):
            assert result['input_ids'] == expected['input_ids'].flatten().tolist()
            assert torch.equal(torch.cat([item['pixel_values'] for item in result['multimodal']]),
                               expected['pixel_values'].to(model.mm_feature_dtype))
            assert torch.equal(torch.stack([item['image_grid_thw'] for item in result['multimodal']]),
                               expected['image_grid_thw'])
        for cached, original in zip(warm['multimodal'], cold['multimodal']):
            assert cached['pixel_values'] is original['pixel_values']
            start, end = cached['offset']
            assert warm['input_ids'][start:end] == [model.image_token_id] * (end - start)

    @pytest.mark.parametrize('raw', [False, True])
    @pytest.mark.parametrize('mixed', [False, True])
    def test_shared_video_options_keep_processor_contract(self, real_model, monkeypatch, raw, mixed):
        from copy import deepcopy

        import numpy as np
        import torch
        from PIL import Image

        model = real_model
        y, x = np.indices((112, 112))
        image = Image.fromarray(np.stack([x * 2, y * 2, (x + y) % 256], axis=-1).astype(np.uint8))
        frames = np.stack([np.asarray(image)] * 4)
        metadata = dict(total_num_frames=64, fps=8.0, duration=8.0, frames_indices=[0, 21, 42, 63])
        before_metadata = deepcopy(metadata)
        items = ([dict(type='image', data=image)] if mixed else []) + [
            dict(type='video', data=frames, video_metadata=metadata)]
        prompt = (([1, model.image_token_id, 2] if mixed else [1]) + [model.input_video_token_id, 3]
                  if raw else (f'<|begin_of_image|>{model.image_token}<|end_of_image|>' if mixed else '')
                  + f'<|begin_of_video|>{model.video_token}<|end_of_video|>')
        budget = {'min_image_tokens': 64, 'max_image_tokens': 64}
        options = {'image': {**budget, 'return_tensors': 'np'},
                   'video': {**budget, 'return_metadata': False, 'return_tensors': 'np',
                             'fps': 99, 'do_sample_frames': True, 'video_metadata': [{'fps': 999}]}}
        before_options = deepcopy(options)
        defaults = [(p.patch_size, p.temporal_patch_size, p.merge_size) for p in
                    (model.processor.image_processor, model.processor.video_processor)]
        sample = Mock(side_effect=AssertionError('Frames must not be sampled again'))
        monkeypatch.setattr(model.processor.video_processor, 'sample_frames', sample)
        frame_builder = Mock(wraps=model.processor.replace_frame_token_id)
        monkeypatch.setattr(model.processor, 'replace_frame_token_id', frame_builder)
        processor_texts = []
        processor_call = type(model.processor).__call__

        def track_text(processor, *args, **kwargs):
            processor_texts.append(kwargs.get('text'))
            return processor_call(processor, *args, **kwargs)

        monkeypatch.setattr(type(model.processor), '__call__', track_text)
        result = model.preprocess([dict(role='user', content=items)], prompt, options)
        assert processor_texts == [None if raw else [prompt]]
        timestamps = [call.args[0] for call in frame_builder.call_args_list]
        assert timestamps and timestamps == [0.0, 5.25] * (len(timestamps) // 2)
        expected_size = budget
        expected = model.processor(text=None if raw else [prompt],
                                   images=[image] if mixed else None, videos=[frames], return_tensors='pt',
                                   images_kwargs=expected_size,
                                   videos_kwargs={**expected_size, 'return_metadata': True,
                                                  'video_metadata': [deepcopy(before_metadata)],
                                                  'do_sample_frames': False, 'do_resize': True})
        for key in ('pixel_values', 'pixel_values_videos'):
            if key in expected:
                actual = torch.cat([item[key] for item in result['multimodal'] if key in item])
                torch.testing.assert_close(actual, expected[key], rtol=0, atol=0)
        if not raw:
            assert result['input_ids'] == expected['input_ids'].flatten().tolist()
        else:
            image_ids = (model.processor.tokenizer.encode(model.processor.replace_image_token(expected, 0),
                                                          add_special_tokens=False) if mixed else [])
            from transformers.video_utils import VideoMetadata

            video_inputs = {**expected, 'video_metadata': [VideoMetadata(**before_metadata)]}
            video_ids = model.processor.tokenizer.encode(model.processor.replace_video_token(video_inputs, 0),
                                                         add_special_tokens=False)
            assert result['input_ids'] == ([1] + image_ids + [2] if mixed else [1]) + video_ids + [3]
        assert options == before_options and metadata == before_metadata
        assert defaults == [(p.patch_size, p.temporal_patch_size, p.merge_size) for p in
                            (model.processor.image_processor, model.processor.video_processor)]
        sample.assert_not_called()

    def test_native_image_overrides_reach_processor(self, model):
        import torch

        from lmdeploy.vl.constants import Modality
        from lmdeploy.vl.model.base import MultimodalSpecialTokens

        model._shared_video_token = False
        model.mm_tokens = MultimodalSpecialTokens(image_token_id=42, video_token_id=43)
        model.processor = Mock(return_value={
            'image_grid_thw': torch.tensor([[1, 4, 4]]),
            'pixel_values': torch.zeros(16, 1),
        })
        model.processor.image_processor = SimpleNamespace(min_image_tokens=16, max_image_tokens=8000, merge_size=2)
        image = object()
        options = {'min_image_tokens': 64, 'max_image_tokens': 64}
        result = model.preprocess(
            [{'role': 'user', 'content': [{'type': Modality.IMAGE, 'data': image}]}],
            [1, 42], {'image': options})

        assert model.processor.call_args.kwargs['images_kwargs'] == options
        assert result['input_ids'] == [1, 42, 42, 42, 42]
        assert model.processor.image_processor.min_image_tokens == 16

    @pytest.mark.parametrize('trust_remote_code', [False, True])
    def test_build_preprocessor(self, monkeypatch, model, trust_remote_code):
        from lmdeploy.vl.model import glm5_next

        processor = SimpleNamespace(image_processor=object(), video_processor=object(),
                                    image_token='<image>', video_token='<video>')
        loader = Mock(return_value=processor)
        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', loader)
        model.build_preprocessor(trust_remote_code=trust_remote_code)
        loader.assert_called_once_with(model.model_path, trust_remote_code=trust_remote_code)
        assert model.processor is processor
        assert model.mm_tokens.image_token_id == 42
        assert model.mm_tokens.video_token_id == 43

    @pytest.mark.parametrize('missing', ['image_processor', 'video_processor'])
    def test_missing_native_processor_fails(self, monkeypatch, model, missing):
        from lmdeploy.vl.model import glm5_next

        components = {key: object() for key in ['image_processor', 'video_processor'] if key != missing}
        processor = SimpleNamespace(**components)
        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', Mock(return_value=processor))
        with pytest.raises(ValueError, match='native image/video processor'):
            model.build_preprocessor()

    @pytest.mark.parametrize('error', [ImportError, ValueError, OSError, RuntimeError])
    def test_load_error_propagates(self, monkeypatch, model, error):
        from lmdeploy.vl.model import glm5_next

        monkeypatch.setattr(glm5_next.AutoProcessor, 'from_pretrained', Mock(side_effect=error('load failed')))
        with pytest.raises(error, match='load failed'):
            model.build_preprocessor()
