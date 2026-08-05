from types import SimpleNamespace

import torch
from PIL import Image

from lmdeploy.vl.model import internvl as internvl_module
from lmdeploy.vl.model.internvl import InternVLVisionModel


def test_internvl_skip_preprocess_bypasses_dynamic_preprocess(monkeypatch):
    def fail_dynamic_preprocess(*args, **kwargs):
        raise AssertionError('dynamic_preprocess should be skipped')

    monkeypatch.setattr(internvl_module, 'dynamic_preprocess', fail_dynamic_preprocess)

    model = InternVLVisionModel.__new__(InternVLVisionModel)
    model.config = SimpleNamespace(min_dynamic_patch=1, max_dynamic_patch=6, use_thumbnail=False)
    model.vision_config = SimpleNamespace(image_size=448)
    model.transform = lambda image: torch.tensor(image.size, dtype=torch.float32)

    image = Image.new('RGB', (320, 240))
    pixel_values = model._preprocess_v1_5(image, params={'skip_preprocess': True})

    assert pixel_values.shape == (1, 2)
    torch.testing.assert_close(pixel_values[0], torch.tensor([320.0, 240.0]))
