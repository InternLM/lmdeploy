# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.hasher import make_multimodal_content_hash


def test_multimodal_content_hash_is_stable_for_nested_values():
    data = {
        'tensor': torch.arange(4, dtype=torch.float32).reshape(2, 2),
        'array': np.arange(3, dtype=np.int64),
        'modality': Modality.IMAGE,
    }
    meta = {'image_token_id': 99, 'shape': [2, 2]}
    mrope_pos_ids = np.arange(6, dtype=np.int64).reshape(2, 3)

    hash1 = make_multimodal_content_hash(data, meta, mrope_pos_ids)
    hash2 = make_multimodal_content_hash(dict(reversed(data.items())), dict(reversed(meta.items())),
                                         mrope_pos_ids.copy())

    assert hash1 == hash2


def test_multimodal_content_hash_changes_with_payload_meta_or_mrope():
    data = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    meta = {'image_token_id': 99}
    mrope_pos_ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    base = make_multimodal_content_hash(data, meta, mrope_pos_ids)

    assert base != make_multimodal_content_hash(data + 1, meta, mrope_pos_ids)
    assert base != make_multimodal_content_hash(data, {'image_token_id': 100}, mrope_pos_ids)
    assert base != make_multimodal_content_hash(data, meta, mrope_pos_ids + 1)
