# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import resource
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLInputProcessor
from lmdeploy.pytorch.multimodal.data_type import get_image_mrope_pos_ids, make_multimodal_content_hash
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.media.image import EncodedImage
from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel
from lmdeploy.vl.model.image_cache import ImageProcessorCache, ProcessedImage


@pytest.fixture
def glm_model():
    class ImageProcessor:
        merge_size = 2
        size = {'shortest_edge': 4, 'longest_edge': 16}
        calls = 0

        def __call__(self, images, return_tensors, **kwargs):
            self.calls += 1
            value = images[0].convert('RGB').getpixel((0, 0))[0]
            return dict(pixel_values=torch.full((4, 3), float(value)),
                        image_grid_thw=torch.tensor([[1, 2, 2]]))

    model = GLM5NextVisionModel.__new__(GLM5NextVisionModel)
    model.backend = 'pytorch'
    model.mm_feature_dtype = torch.bfloat16
    model.hf_config = SimpleNamespace(dtype=torch.bfloat16)
    model.image_token_id = 9
    model.processor = SimpleNamespace(image_processor=ImageProcessor())
    model.image_cache = ImageProcessorCache(1024)
    return model


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


@pytest.mark.parametrize('backend, config, shared', [
    ('pytorch', None, False),
    ('pytorch', PytorchEngineConfig(), False),
    ('pytorch', PytorchEngineConfig(enable_mp_engine=False, mp_engine_backend='ray'), False),
    ('pytorch', PytorchEngineConfig(enable_mp_engine=True, mp_engine_backend='mp'), True),
    ('pytorch', PytorchEngineConfig(enable_mp_engine=True, mp_engine_backend='ray'), False),
    ('pytorch', PytorchEngineConfig(enable_mp_engine=True, distributed_executor_backend='ray'), True),
    ('turbomind', TurbomindEngineConfig(), False),
])
def test_image_encoder_selects_shared_storage_by_transport(monkeypatch, glm_model, backend, config, shared):
    from lmdeploy.vl import engine

    monkeypatch.setattr(engine, 'load_vl_model', lambda *args, **kwargs: glm_model)
    monkeypatch.setattr(engine, 'is_bf16_supported', lambda device: True)
    monkeypatch.setattr(torch.multiprocessing, 'get_sharing_strategy', lambda: 'file_descriptor')
    if not shared:
        monkeypatch.setattr(torch.Tensor, 'share_memory_',
                            Mock(side_effect=AssertionError('unexpected shared storage')))
    encoder = engine.ImageEncoder('unused', backend, VisionConfig(image_cache_max_bytes=1024), backend_config=config)
    try:
        images = [Image.new('RGB', (2, 2), (1, 0, 0))]
        cold = glm_model._preprocess_images(images, [9], None)
        warm = glm_model._preprocess_images(images, [3, 9], None)
        assert glm_model.processor.image_processor.calls == 1
        assert warm['multimodal'][0]['pixel_values'] is cold['multimodal'][0]['pixel_values']
        assert warm['multimodal'][0]['pixel_values'].is_shared() == shared
        assert warm['multimodal'][0]['offset'] == (1, 2)
    finally:
        encoder.executor.shutdown()


def test_image_cache_skips_shared_storage_for_file_system_strategy(monkeypatch, glm_model):
    glm_model.image_cache.share_memory = True
    monkeypatch.setattr(torch.multiprocessing, 'get_sharing_strategy', lambda: 'file_system')
    monkeypatch.setattr(torch.Tensor, 'share_memory_', Mock(side_effect=AssertionError('unexpected shared storage')))
    output = glm_model._preprocess_images([Image.new('RGB', (2, 2))], [9], None)
    assert not output['multimodal'][0]['pixel_values'].is_shared()


@pytest.mark.parametrize('encoded', [False, True])
@pytest.mark.parametrize('share_memory', [False, True])
def test_disabled_image_cache_skips_key_but_preserves_content_hash(monkeypatch, glm_model, encoded, share_memory):
    glm_model.image_cache = ImageProcessorCache(0, share_memory=share_memory)
    monkeypatch.setattr(glm_model.image_cache, 'key', Mock(side_effect=AssertionError('unexpected cache key')))
    image = Image.new('RGB', (2, 2), (7, 0, 0))
    if encoded:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image = EncodedImage(buffer.getvalue(), 'RGB')
    first = glm_model._preprocess_images([image], [9], None)['multimodal'][0]
    second = glm_model._preprocess_images([image], [3, 9], None)['multimodal'][0]
    assert glm_model.processor.image_processor.calls == 2
    assert glm_model.image_cache.current_bytes == 0
    assert first['pixel_values'] is not second['pixel_values']
    assert torch.equal(first['pixel_values'], second['pixel_values'])
    assert first['content_hash'] == second['content_hash'] == make_multimodal_content_hash(
        first['pixel_values'], dict(grid_thw=first['image_grid_thw'], image_token_id=9),
        get_image_mrope_pos_ids(first['image_grid_thw']))
    assert first['pixel_values'].is_shared() == share_memory


def test_image_cache_process_reuses_encoded_images_without_decoding(monkeypatch):
    image = EncodedImage(b'encoded image', 'RGB')
    decoded = Image.new('RGB', (2, 2))
    decode = Mock(return_value=decoded)
    monkeypatch.setattr(EncodedImage, 'decode', decode)
    features = ProcessedImage(torch.ones(4), torch.tensor([1, 2, 2]), 'content')
    preprocess = Mock(return_value=features)
    cache = ImageProcessorCache(1024)
    size = {'longest_edge': 16}
    assert cache.process([image, image], preprocess, torch.float32, size) == [features, features]
    assert cache.process([image], preprocess, torch.float32, size) == [features]
    decode.assert_called_once_with()
    preprocess.assert_called_once_with(decoded, size)


@pytest.mark.parametrize('error', [
    'unable to allocate shared memory(shm): No space left on device (28)',
    'unable to allocate shared memory(shm): Success (0)',
    'unexpected allocation failure',
])
@pytest.mark.parametrize('max_bytes', [0, 1024])
def test_glm_images_handle_shared_memory_exhaustion(monkeypatch, glm_model, error, max_bytes):
    model = glm_model
    model.image_cache = ImageProcessorCache(max_bytes, share_memory=True)
    # Allocate before patching so PyTorch internals cannot affect the call count.
    shared_pixels = [torch.empty((4, 3), dtype=torch.bfloat16).share_memory_() for _ in range(2)]
    allocations = 0

    def limited_shm(tensor):
        nonlocal allocations
        allocations += 1
        if allocations >= 3:
            raise RuntimeError(error)
        shared = shared_pixels[allocations - 1]
        shared.copy_(tensor)
        return tensor.set_(shared)

    monkeypatch.setattr(torch.Tensor, 'share_memory_', limited_shm)
    images = [Image.new('RGB', (2, 2), (value, 0, 0)) for value in range(1, 5)]
    if error == 'unexpected allocation failure':
        with pytest.raises(RuntimeError, match=error):
            model._preprocess_images(images, [9] * 4, None)
        return
    output = model._preprocess_images(images, [9] * 4, None)
    assert allocations == 3  # Do not retry a full shm for every remaining image.
    assert model.image_cache.current_bytes <= max_bytes
    assert output['input_ids'] == [9] * 4
    assert [item['pixel_values'].is_shared() for item in output['multimodal']] == [True, True, False, False]
    for value, item in enumerate(output['multimodal'], 1):
        assert torch.all(item['pixel_values'] == value)
        assert item['content_hash'] == make_multimodal_content_hash(
            item['pixel_values'], dict(grid_thw=item['image_grid_thw'], image_token_id=9),
            get_image_mrope_pos_ids(item['image_grid_thw']))


@pytest.mark.parametrize('palette', [False, True])
@pytest.mark.parametrize('share_memory', [False, True])
def test_glm_image_cache_reuses_features_and_keeps_request_offsets(glm_model, palette, share_memory):
    def image(value):
        if palette:
            result = Image.new('P', (2, 2))
            result.putpalette([value, 0, 0] + [0] * 765)
            return result
        buffer = BytesIO()
        Image.new('RGB', (2, 2), (value, 0, 0)).save(buffer, format='PNG')
        return EncodedImage(buffer.getvalue(), 'RGB')

    model = glm_model
    model.image_cache.share_memory = share_memory

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
        assert item.data.is_shared() == share_memory
        assert item.data.untyped_storage().nbytes() == item.data.numel() * item.data.element_size()
    run([image(3)], [9])
    run([image(1)], [9], image={'max_pixels': 8})
    model.mm_feature_dtype = torch.float32
    run([image(1)], [9])
    assert model.processor.image_processor.calls == 5
