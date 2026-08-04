# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.hasher import make_multimodal_content_hash
from lmdeploy.vl.model.preprocess_utils import attach_multimodal_content_hashes, get_expanded_mm_items


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


def test_attach_multimodal_content_hashes_to_single_image():
    items = {
        Modality.IMAGE: {
            'feature': torch.arange(2, dtype=torch.float32).reshape(2, 1),
            'image_grid_thw': torch.tensor([[1, 2, 1]]),
            'offset': [(0, 2)],
        }
    }
    expanded = get_expanded_mm_items(items, _Tokens())

    attach_multimodal_content_hashes(expanded)

    assert len(expanded[0]['content_hash']) == 64
    content_view = {key: value for key, value in expanded[0].items() if key not in ('content_hash', 'offset')}
    assert expanded[0]['content_hash'] == make_multimodal_content_hash(content_view)


def test_attach_multimodal_content_hashes_ignores_offset():
    item1 = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.arange(2, dtype=torch.float32).reshape(2, 1),
        'image_grid_thw': torch.tensor([1, 2, 1]),
        'offset': (0, 2),
        'image_token_id': 42,
    }
    item2 = dict(item1, offset=(10, 12))

    attach_multimodal_content_hashes([item1, item2])

    assert item1['content_hash'] == item2['content_hash']


def test_attach_multimodal_content_hashes_changes_with_processed_content():
    item1 = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.arange(2, dtype=torch.float32).reshape(2, 1),
        'image_grid_thw': torch.tensor([1, 2, 1]),
        'offset': (0, 2),
        'image_token_id': 42,
    }
    item2 = dict(item1, pixel_values=item1['pixel_values'] + 1)

    attach_multimodal_content_hashes([item1, item2])

    assert item1['content_hash'] != item2['content_hash']


def test_attach_multimodal_content_hashes_to_frame_split_videos():
    items = {
        Modality.VIDEO: {
            'feature': torch.arange(8, dtype=torch.float32).reshape(8, 1),
            'video_grid_thw': torch.tensor([[2, 2, 1], [2, 2, 1]]),
            'offset': [(0, 1), (1, 2), (2, 3), (3, 4)],
        }
    }
    expanded = get_expanded_mm_items(items, _Tokens())

    attach_multimodal_content_hashes(expanded)

    assert len({entry['content_hash'] for entry in expanded}) == len(expanded)


def test_attach_multimodal_content_hashes_preserves_prompt_order_for_mixed_modalities():
    items = {
        Modality.AUDIO: {
            'feature': torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3),
            'feature_attention_mask': torch.ones(1, 3, dtype=torch.long),
            'offset': [(5, 8)],
        },
        Modality.IMAGE: {
            'feature': torch.arange(2, dtype=torch.float32).reshape(2, 1),
            'image_grid_thw': torch.tensor([[1, 2, 1]]),
            'offset': [(0, 2)],
        },
    }
    expanded = get_expanded_mm_items(items, _Tokens())

    attach_multimodal_content_hashes(expanded)

    assert [entry['modality'] for entry in expanded] == [Modality.IMAGE, Modality.AUDIO]
    assert all(len(entry['content_hash']) == 64 for entry in expanded)
