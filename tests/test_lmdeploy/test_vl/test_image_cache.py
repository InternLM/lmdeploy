# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import resource
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLInputProcessor
from lmdeploy.pytorch.multimodal.data_type import make_multimodal_content_hash
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.media.image import EncodedImage
from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel
from lmdeploy.vl.model.image_cache import ImageProcessorCache, ProcessedImage


def test_image_cache_budget_and_inflight_ownership():
    def entry(value):
        return ProcessedImage(torch.full((4, 3), value), torch.tensor([1, 2, 2]), str(value))

    first, second, third = entry(1.), entry(2.), entry(3.)
    cache = ImageProcessorCache(first.nbytes * 2)
    cache.put(('a',), first)
    cache.put(('b',), second)
    inflight = cache.get(('a',))
    cache.put(('c',), third)
    assert cache.get(('b',)) is None
    assert cache.get(('a',)) is first
    assert cache.current_bytes == first.nbytes * 2
    cache.put(('d',), second)
    cache.put(('e',), third)
    assert cache.get(('a',)) is None
    assert torch.all(inflight.pixel_values == 1)
    oversized = ProcessedImage(torch.zeros(1024), first.image_grid_thw, '')
    cache.put(('large',), oversized)
    assert cache.get(('large',)) is None
    assert cache.current_bytes <= cache.max_bytes


def _check_image_cache_shared_fds():
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    initial_fds = len(os.listdir('/proc/self/fd'))
    cache = ImageProcessorCache(2 * 1024**3)
    inflight = ProcessedImage(torch.ones(16).share_memory_(), torch.tensor([1, 2, 2]), 'first')
    cache.put((0,), inflight)
    for index in range(1, 1024):
        cache.put((index,), ProcessedImage(torch.empty(16).share_memory_(), torch.tensor([1, 2, 2]), str(index)))
    assert cache.get((0,)) is None
    assert cache.get((1023,)) is not None
    assert torch.all(inflight.pixel_values == 1)
    assert len(os.listdir('/proc/self/fd')) <= initial_fds + 32 + 1
    del cache
    assert len(os.listdir('/proc/self/fd')) == initial_fds + 1
    del inflight
    assert len(os.listdir('/proc/self/fd')) == initial_fds


def test_image_cache_bounds_retained_shared_fds():
    # Isolate the process-wide FD limit and PyTorch sharing strategy.
    process = mp.get_context('spawn').Process(target=_check_image_cache_shared_fds, daemon=True)
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    process.close()


def test_palette_image_cache_key_includes_transparency():
    image = Image.new('P', (2, 2))
    image.putpalette([255, 0, 0] + [0] * 765)
    opaque_key = ImageProcessorCache.key(image, torch.bfloat16, None)
    transparent = image.copy()
    transparent.info['transparency'] = 0
    assert transparent.tobytes() == image.tobytes()
    assert ImageProcessorCache.key(transparent, torch.bfloat16, None) != opaque_key
    assert ImageProcessorCache.key(image.copy(), torch.bfloat16, None) == opaque_key


@pytest.mark.parametrize('palette', [False, True])
def test_glm_image_cache_reuses_features_and_keeps_request_offsets(palette):
    class ImageProcessor:
        merge_size = 2
        size = {'shortest_edge': 4, 'longest_edge': 16}
        calls = 0

        def __call__(self, images, return_tensors, **kwargs):
            self.calls += 1
            value = images[0].convert('RGB').getpixel((0, 0))[0]
            return dict(pixel_values=torch.full((4, 3), float(value)),
                        image_grid_thw=torch.tensor([[1, 2, 2]]))

    def image(value):
        if palette:
            result = Image.new('P', (2, 2))
            result.putpalette([value, 0, 0] + [0] * 765)
            return result
        buffer = BytesIO()
        Image.new('RGB', (2, 2), (value, 0, 0)).save(buffer, format='PNG')
        return EncodedImage(buffer.getvalue(), 'RGB')

    model = GLM5NextVisionModel.__new__(GLM5NextVisionModel)
    model.backend = 'pytorch'
    model.mm_feature_dtype = torch.bfloat16
    model.image_token_id = 9
    model.processor = SimpleNamespace(image_processor=ImageProcessor())
    model.image_cache = ImageProcessorCache(1024)

    def run(images, prompt, **kwargs):
        messages = [dict(content=[dict(type=Modality.IMAGE, data=x) for x in images])]
        return model.preprocess(messages, prompt, kwargs or None)

    cold = run([image(1), image(2)], [9, 3, 9])
    warm = run([image(2), image(1)], [4, 9, 9])
    assert model.processor.image_processor.calls == 2
    assert cold['multimodal'][0]['content_hash'] != cold['multimodal'][1]['content_hash']
    assert warm['input_ids'] == [4, 9, 9]
    assert warm['multimodal'][0]['offset'] == (1, 2)
    assert warm['multimodal'][0]['pixel_values'] is cold['multimodal'][1]['pixel_values']
    inputs = Qwen3VLInputProcessor(None, torch.bfloat16).preprocess_input(**{
        'input_ids': warm['input_ids'], 'input_multimodals': warm['multimodal']})
    for item in inputs.input_multimodals['mm_data']:
        assert item.content_hash == make_multimodal_content_hash(item.data, item.meta, item.mrope_pos_ids)
        assert item.data.is_shared()
        assert item.data.untyped_storage().nbytes() == item.data.numel() * item.data.element_size()
    run([image(3)], [9])
    run([image(1)], [9], image={'max_pixels': 8})
    model.mm_feature_dtype = torch.float32
    run([image(1)], [9])
    assert model.processor.image_processor.calls == 5
