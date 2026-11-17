# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the native GLM processor, including per-request budget
isolation."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lmdeploy.vl.model.base import MultimodalSpecialTokens
from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel


@pytest.mark.parametrize('max_pixels', [25088, 100352, 401408])
def test_native_pixel_budget_and_request_isolation(max_pixels):
    native = pytest.importorskip('transformers.models.glm5_next.image_processing_glm5_next')
    image_processor = native.Glm5NextImageProcessor()

    class Processor:
        def __init__(self):
            self.image_processor = image_processor

        def __call__(self, images, **kwargs):
            result = self.image_processor(images=images, return_tensors='pt', **kwargs.get('images_kwargs', {}))
            count = int(result['image_grid_thw'].prod()) // image_processor.merge_size**2
            result['input_ids'] = torch.full((1, count), 42)
            return result

    model = GLM5NextVisionModel.__new__(GLM5NextVisionModel)
    model.processor = Processor()
    model._shared_video_token = False
    model.mm_tokens = MultimodalSpecialTokens(image_token='<image>', image_token_id=42)
    model.hf_config = SimpleNamespace(image_token_id=42)
    messages = [dict(role='user', content=[dict(type='image', data=Image.new('RGB', (1024, 1024)))])]
    before = (image_processor.min_image_tokens, image_processor.max_image_tokens)
    # Match the existing GLM-4V compatibility path's spatiotemporal pixel contract.
    pixels_per_token = image_processor.temporal_patch_size * (
        image_processor.patch_size * image_processor.merge_size)**2
    limited = model.preprocess(messages, '<image>', {'image': {'min_pixels': 25088, 'max_pixels': max_pixels}})
    assert 0 < limited['input_ids'].count(42) <= max_pixels // pixels_per_token
    default = model.preprocess(messages, '<image>')
    assert default['input_ids'].count(42) == 1369
    assert (image_processor.min_image_tokens, image_processor.max_image_tokens) == before
