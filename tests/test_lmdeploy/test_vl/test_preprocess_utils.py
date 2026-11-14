# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.preprocess_utils import get_expanded_mm_items, get_override_size, get_processor_kwargs


def test_processor_kwargs_preserve_pixel_aliases_and_copy_options():
    processor = SimpleNamespace(size={'shortest_edge': 64, 'longest_edge': 1024})
    options = {'max_pixels': 256, 'do_normalize': False}
    assert get_processor_kwargs(processor, options, 'image') == {
        'size': {'shortest_edge': 64, 'longest_edge': 256}, 'do_normalize': False}
    assert options == {'max_pixels': 256, 'do_normalize': False}
    options = {'size': {'shortest_edge': 64, 'longest_edge': 256}}
    output = get_processor_kwargs(processor, options, 'image')
    output['size']['shortest_edge'] = 1
    assert options['size']['shortest_edge'] == 64


@pytest.mark.parametrize('options', [
    {'min_image_tokens': 64, 'max_image_tokens': 64},
    {'min_image_tokens': 64}, {'max_image_tokens': 64},
])
def test_processor_kwargs_native_token_budgets(options):
    processor = SimpleNamespace(min_image_tokens=16, max_image_tokens=8000)
    assert get_processor_kwargs(processor, options, 'image') == options
    assert vars(processor) == {'min_image_tokens': 16, 'max_image_tokens': 8000}


@pytest.mark.parametrize('options', [
    {'min_pixels': 64}, {'max_pixels': 64}, {'size': {'longest_edge': 64}},
    {'min_image_tokens': 0}, {'max_image_tokens': True}, {'max_image_tokens': None},
    {'min_image_tokens': 64, 'max_image_tokens': 16},
])
def test_processor_kwargs_reject_ambiguous_or_invalid_native_budgets(options):
    processor = SimpleNamespace(min_image_tokens=16, max_image_tokens=8000)
    with pytest.raises(ValueError, match='token'):
        get_processor_kwargs(processor, options, 'image')


def test_processor_kwargs_missing_pixel_defaults_and_unsupported_tokens():
    processor = SimpleNamespace(size={'shortest_edge': None, 'longest_edge': 1})
    with pytest.raises(ValueError, match='pixel-size defaults'):
        get_override_size(processor, {'max_pixels': 64}, 'image')
    with pytest.raises(ValueError, match='does not support'):
        get_processor_kwargs(processor, {'min_image_tokens': 64}, 'image')


def test_processor_kwargs_reject_non_object_options():
    processor = SimpleNamespace(size={'shortest_edge': 64, 'longest_edge': 1024})
    with pytest.raises(ValueError):
        get_processor_kwargs(processor, [], 'image')


@pytest.mark.parametrize('options', [
    {'max_pixels': 256, 'size': {'shortest_edge': 1, 'longest_edge': 1}},
    {'min_pixels': 512, 'max_pixels': 256, 'size': {}},
    {'max_pixels': 0}, {'min_pixels': True},
])
def test_processor_kwargs_preserve_legacy_pixel_alias_semantics(options):
    processor = SimpleNamespace(size={'shortest_edge': 64, 'longest_edge': 1024})
    size = get_override_size(processor, options, 'image')
    assert get_processor_kwargs(processor, options, 'image') == ({} if size is None else {'size': size})


def test_processor_kwargs_ignore_loader_owned_options_without_mutating_request(monkeypatch):
    warning = Mock()
    monkeypatch.setattr('lmdeploy.vl.model.preprocess_utils.logger.warning', warning)
    options = {'fps': 1, 'num_frames': 2, 'min_frames': 2, 'max_frames': 8,
               'do_sample_frames': True, 'video_metadata': [{'fps': 999}], 'max_pixels': 256}
    processor = SimpleNamespace(size={'shortest_edge': 64, 'longest_edge': 1024})
    output = get_processor_kwargs(processor, options, 'video')
    assert output == {'size': {'shortest_edge': 64, 'longest_edge': 256}}
    assert options['fps'] == 1 and options['video_metadata'] == [{'fps': 999}]
    warning.assert_called_once()
    assert 'Ignoring loader-owned' in warning.call_args.args[0]
    assert 'fps' in warning.call_args.args[1]


def test_processor_kwargs_real_native_image_is_request_local():
    module = pytest.importorskip('transformers.models.glm5_next.image_processing_glm5_next')
    from PIL import Image

    processor = module.Glm5NextImageProcessor()
    image = Image.new('RGB', (112, 112), (120, 30, 200))
    default = processor(images=[image], return_tensors='pt')
    custom = processor(images=[image], return_tensors='pt', **get_processor_kwargs(
        processor, {'min_image_tokens': 64, 'max_image_tokens': 64}, 'image'))
    after = processor(images=[image], return_tensors='pt')
    assert custom['image_grid_thw'].tolist() == [[1, 16, 16]]
    assert default['image_grid_thw'].tolist() == [[1, 8, 8]]
    assert torch.equal(after['pixel_values'], default['pixel_values'])


def test_processor_kwargs_real_size_processor_preserves_pixel_budget():
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor()
    image = Image.new('RGB', (112, 112))
    kwargs = get_processor_kwargs(processor, {'min_pixels': 1024, 'max_pixels': 4096}, 'image')
    custom = processor(images=[image], return_tensors='pt', **kwargs)
    reference = processor(images=[image], return_tensors='pt', size={'shortest_edge': 1024, 'longest_edge': 4096})
    assert torch.equal(custom['pixel_values'], reference['pixel_values'])
    assert torch.equal(custom['image_grid_thw'], reference['image_grid_thw'])



class _Tokens:

    image_token_id = 42
    video_token_id = 43
    audio_token_id = 44

    def get_token_id_by_modality(self, modality):
        if modality == Modality.IMAGE:
            return self.image_token_id
        if modality == Modality.VIDEO:
            return self.video_token_id
        if modality == Modality.AUDIO:
            return self.audio_token_id
        raise AssertionError(f'unexpected modality: {modality}')


def _assert_compact_storage(tensor):
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_expand_bundled_image_items_use_compact_tensor_storage():
    feature = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    items = {
        Modality.IMAGE: {
            'feature': feature,
            'image_grid_thw': torch.tensor([[1, 2, 1], [1, 2, 1], [1, 2, 1]]),
            'offset': [(0, 1), (1, 2), (2, 3)],
        }
    }

    expanded = get_expanded_mm_items(items, _Tokens())

    assert len(expanded) == 3
    for entry in expanded:
        _assert_compact_storage(entry['pixel_values'])


def test_expand_qwen3vl_frame_video_items_use_compact_tensor_storage():
    feature = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    items = {
        Modality.VIDEO: {
            'feature': feature,
            'video_grid_thw': torch.tensor([[2, 2, 1], [2, 2, 1]]),
            'offset': [(0, 1), (1, 2), (2, 3), (3, 4)],
        }
    }

    expanded = get_expanded_mm_items(items, _Tokens())

    assert len(expanded) == 4
    for entry in expanded:
        _assert_compact_storage(entry['pixel_values_videos'])


def test_expand_qwen3_omni_whole_video_items_use_compact_tensor_storage():
    feature = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    items = {
        Modality.VIDEO: {
            'feature': feature,
            'video_grid_thw': torch.tensor([[2, 2, 1], [2, 2, 1]]),
            'video_second_per_grid': torch.tensor([1.0, 2.0]),
            'offset': [(0, 2), (2, 4)],
        }
    }

    expanded = get_expanded_mm_items(items, _Tokens())

    assert len(expanded) == 2
    assert [entry['video_grid_thw'].tolist() for entry in expanded] == [[2, 2, 1], [2, 2, 1]]
    assert [entry['second_per_grid'] for entry in expanded] == [1.0, 2.0]
    for entry in expanded:
        _assert_compact_storage(entry['pixel_values_videos'])


def test_expand_audio_items_use_compact_tensor_storage():
    feature = torch.arange(2 * 128 * 300, dtype=torch.float32).reshape(2, 128, 300)
    mask = torch.ones(2, 300, dtype=torch.long)
    items = {
        Modality.AUDIO: {
            'feature': feature,
            'feature_attention_mask': mask,
            'offset': [(0, 39), (39, 78)],
        }
    }

    expanded = get_expanded_mm_items(items, _Tokens())

    assert len(expanded) == 2
    for entry in expanded:
        _assert_compact_storage(entry['input_features'])
        _assert_compact_storage(entry['feature_attention_mask'])
