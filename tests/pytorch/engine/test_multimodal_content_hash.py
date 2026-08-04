# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLInputProcessor
from lmdeploy.vl.constants import Modality


def test_qwen3vl_input_processor_preserves_image_content_hash():
    processor = Qwen3VLInputProcessor(config=object(), dtype=torch.float32)
    result = processor.preprocess_input(
        input_ids=[1, 2, 3],
        input_multimodals=[{
            'modality': Modality.IMAGE,
            'pixel_values': torch.ones(1, 1),
            'image_grid_thw': torch.tensor([1, 2, 2]),
            'offset': (1, 2),
            'image_token_id': 99,
            'content_hash': 'image-a',
        }],
    )

    assert result.input_multimodals['mm_data'][0].content_hash == 'image-a'


def test_qwen3vl_input_processor_preserves_video_content_hash():
    processor = Qwen3VLInputProcessor(config=object(), dtype=torch.float32)
    result = processor.preprocess_input(
        input_ids=[1, 2, 3],
        input_multimodals=[{
            'modality': Modality.VIDEO,
            'pixel_values_videos': torch.ones(1, 1),
            'video_grid_thw': torch.tensor([1, 2, 2]),
            'offset': (1, 2),
            'video_token_id': 99,
            'content_hash': 'video-a',
        }],
    )

    assert result.input_multimodals['mm_data'][0].content_hash == 'video-a'
