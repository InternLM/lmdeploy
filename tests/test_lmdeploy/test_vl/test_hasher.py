# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import numpy as np
import torch

from lmdeploy.vl import hasher as mm_hasher
from lmdeploy.vl.constants import Modality


@dataclass
class DummyModalData:
    data: object
    meta: dict | None = None
    mrope_pos_ids: np.ndarray | None = None
    content_hash: str | None = None


def test_multimodal_content_hash_is_stable_for_nested_values():
    data = {
        'tensor': torch.arange(4, dtype=torch.float32).reshape(2, 2),
        'array': np.arange(3, dtype=np.int64),
        'modality': Modality.IMAGE,
    }
    meta = {'image_token_id': 99, 'shape': [2, 2]}
    mrope_pos_ids = np.arange(6, dtype=np.int64).reshape(2, 3)

    hash1 = mm_hasher.make_multimodal_content_hash(data, meta, mrope_pos_ids)
    hash2 = mm_hasher.make_multimodal_content_hash(dict(reversed(data.items())), dict(reversed(meta.items())),
                                                   mrope_pos_ids.copy())

    assert hash1 == hash2


def test_multimodal_content_hash_changes_with_payload_meta_or_mrope():
    data = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    meta = {'image_token_id': 99}
    mrope_pos_ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    base = mm_hasher.make_multimodal_content_hash(data, meta, mrope_pos_ids)

    assert base != mm_hasher.make_multimodal_content_hash(data + 1, meta, mrope_pos_ids)
    assert base != mm_hasher.make_multimodal_content_hash(data, {'image_token_id': 100}, mrope_pos_ids)
    assert base != mm_hasher.make_multimodal_content_hash(data, meta, mrope_pos_ids + 1)


def test_ensure_multimodal_content_hashes_preserves_existing_hash():
    modal_data = DummyModalData(data=torch.ones(2, 2), meta={'image_token_id': 99}, content_hash='preset')
    input_mms = {'image': [modal_data]}

    assert mm_hasher.ensure_multimodal_content_hashes(input_mms) is input_mms
    assert modal_data.content_hash == 'preset'


def test_ensure_multimodal_content_hashes_populates_missing_hash():
    modal_data = DummyModalData(data=torch.ones(2, 2), meta={'image_token_id': 99})
    mm_hasher.ensure_multimodal_content_hashes({'image': [modal_data]})

    assert isinstance(modal_data.content_hash, str)
    assert len(modal_data.content_hash) == 64


def test_multimodal_item_hash_ignores_position_keys():
    item = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.arange(4, dtype=torch.float32).reshape(2, 2),
        'image_grid_thw': torch.tensor([1, 2, 2]),
        'image_token_id': 99,
        'offset': torch.tensor([4, 8]),
        'start': 4,
        'end': 8,
        'token_begin': 4,
        'token_end': 8,
    }
    moved_item = dict(item, offset=torch.tensor([12, 16]), start=12, end=16, token_begin=12, token_end=16)

    assert mm_hasher.make_multimodal_item_content_hash(item) == mm_hasher.make_multimodal_item_content_hash(moved_item)


def test_multimodal_item_hash_includes_content_keys():
    item = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.arange(4, dtype=torch.float32).reshape(2, 2),
        'image_grid_thw': torch.tensor([1, 2, 2]),
        'image_token_id': 99,
        'offset': torch.tensor([4, 8]),
    }
    changed = dict(item, pixel_values=item['pixel_values'] + 1)

    assert mm_hasher.make_multimodal_item_content_hash(item) != mm_hasher.make_multimodal_item_content_hash(changed)


def test_ensure_multimodal_item_content_hashes_preserves_existing_hash():
    item = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.ones(2, 2),
        'content_hash': 'preset',
    }
    items = [item]

    assert mm_hasher.ensure_multimodal_item_content_hashes(items) is items
    assert item['content_hash'] == 'preset'


def test_ensure_multimodal_item_content_hashes_populates_missing_hash():
    item = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.ones(2, 2),
    }
    items = [item]

    assert mm_hasher.ensure_multimodal_item_content_hashes(items) is items
    assert isinstance(item['content_hash'], str)
    assert len(item['content_hash']) == 64
