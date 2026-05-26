# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.preprocess_utils import get_expanded_mm_items


class _Tokens:

    image_token_id = 42
    video_token_id = 43

    def get_token_id_by_modality(self, modality):
        if modality == Modality.IMAGE:
            return self.image_token_id
        if modality == Modality.VIDEO:
            return self.video_token_id
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


def test_expand_bundled_video_items_use_compact_tensor_storage():
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
