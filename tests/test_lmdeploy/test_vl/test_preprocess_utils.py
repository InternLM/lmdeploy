# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.preprocess_utils import get_expanded_mm_items


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
