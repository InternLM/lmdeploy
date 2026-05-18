from types import SimpleNamespace

import torch

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VisionModel, _postprocess_mm_output


class _Tokens:

    image_token_id = 42
    video_token_id = 43

    def get_token_id_by_modality(self, modality):
        if modality == Modality.IMAGE:
            return self.image_token_id
        if modality == Modality.VIDEO:
            return self.video_token_id
        raise AssertionError(f'unexpected modality: {modality}')


class _DummyImageProcessor:

    merge_size = 1
    size = {'shortest_edge': 1, 'longest_edge': 4}


class _DummyProcessor:

    image_processor = _DummyImageProcessor()

    def __call__(self, **kwargs):
        self.last_output = {
            'input_ids': torch.tensor([[42, 7]], dtype=torch.long),
            'pixel_values': torch.ones((2, 3), dtype=torch.float32),
            'image_grid_thw': torch.tensor([[1, 2, 1]], dtype=torch.long),
        }
        return self.last_output


class _DummyVisionModel(VisionModel):

    def __init__(self):
        super().__init__('dummy', hf_config=SimpleNamespace(pad_token_id=0), backend='pytorch')
        self.processor = _DummyProcessor()
        self.mm_tokens = _Tokens()

    def build_preprocessor(self, trust_remote_code: bool = False):
        pass


def test_postprocess_mm_output_casts_only_floating_tensors():
    payload = {
        'feature': [torch.ones((2, 2), dtype=torch.float32)],
        'meta': {
            'ids': torch.ones(2, dtype=torch.long),
            'text': 'kept',
        },
    }

    converted = _postprocess_mm_output(payload, torch.bfloat16)

    assert converted['feature'][0].dtype is torch.bfloat16
    assert converted['meta']['ids'].dtype is torch.long
    assert converted['meta']['text'] == 'kept'


def test_vision_model_preprocess_casts_feature_to_configured_dtype():
    model = _DummyVisionModel()
    model.set_mm_feature_dtype(torch.bfloat16)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'data': object()}]}]

    result = model.preprocess(messages, input_prompt='dummy')
    item = result['multimodal'][0]

    assert item['pixel_values'].dtype is torch.bfloat16
    assert model.processor.last_output['pixel_values'].dtype is torch.bfloat16
    assert item['image_grid_thw'].dtype is torch.long
    assert result['input_ids'] == [42, 7]


def test_image_encoder_sets_mm_feature_dtype_from_init_argument(monkeypatch):
    from lmdeploy.vl import engine as vl_engine

    dummy_model = _DummyVisionModel()

    def fake_load_vl_model(model_path, backend, backend_config=None, trust_remote_code=False):
        return dummy_model

    monkeypatch.setattr(vl_engine, 'load_vl_model', fake_load_vl_model)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)

    vl_engine.ImageEncoder('dummy', 'pytorch', mm_feature_dtype=torch.bfloat16)

    assert dummy_model.mm_feature_dtype is torch.bfloat16


def test_mp_engine_exposes_resolved_model_config():
    from lmdeploy.pytorch.engine.mp_engine.base import MPEngine

    class DummyMPEngine(MPEngine):

        def _collective_rpc(self, func, *args, **kwargs):
            if func == 'get_engine_config':
                return SimpleNamespace(max_batch_size=1)
            if func == 'get_model_config':
                return SimpleNamespace(dtype=torch.bfloat16)
            raise AssertionError(f'unexpected rpc: {func}')

    engine = DummyMPEngine()

    assert engine.model_config.dtype is torch.bfloat16


def test_mp_worker_exposes_resolved_model_config():
    from lmdeploy.pytorch.engine.mp_engine.base_worker import EngineWorkerBase

    model_config = SimpleNamespace(dtype=torch.bfloat16)
    worker = EngineWorkerBase.__new__(EngineWorkerBase)
    worker.engine = SimpleNamespace(model_config=model_config)

    assert worker.get_model_config() is model_config
