# Copyright (c) OpenMMLab. All rights reserved.
from queue import Queue
from unittest.mock import Mock

from lmdeploy.turbomind.loader import BaseLoader, StateDictLoader, create_loader
from lmdeploy.turbomind.model_loader import ModelLoader


class _FakeModelComm:

    def attn_tp_rank(self, gpu):
        return gpu

    def mlp_tp_rank(self, gpu):
        return gpu

    def model_tp_rank(self, gpu):
        return gpu

    def context(self, gpu):
        return f'ctx:{gpu}'

    def root(self, gpu):
        return f'root:{gpu}'


def test_base_loader_defaults_to_no_layer_pattern():
    class _TestLoader(BaseLoader):
        def items(self):
            raise NotImplementedError

        def all_items(self):
            raise NotImplementedError

    loader = _TestLoader('model-path')

    assert loader.model_path == 'model-path'
    assert loader.pattern is None
    assert loader.mappings == []


def test_state_dict_loader_can_be_created_without_pattern_or_mappings():
    queue = Queue()

    loader = create_loader(queue)

    assert isinstance(loader, StateDictLoader)
    assert loader.pattern is None


def test_model_loader_export_uses_all_params_loader_without_model_metadata(monkeypatch):
    import lmdeploy.turbomind.model_loader as model_loader_mod

    loader = Mock()
    loader.all_items.return_value = {'weight': object()}
    create_loader = Mock(return_value=loader)
    monkeypatch.setattr(model_loader_mod, 'create_loader', create_loader)

    model = Mock()
    model._loader_mappings = []
    model_loader = ModelLoader(model, _FakeModelComm(), 1, 'model-path',
                               data_type=Mock(), engine_config=Mock(attn_tp_size=1, attn_cp_size=1, mlp_tp_size=1))

    model_loader.export()

    create_loader.assert_called_once_with('model-path', None, [])
    model.set_params.assert_called_once_with(loader.all_items.return_value)
    model.model.assert_called_once_with()


def test_model_loader_export_iter_uses_all_params_loader_without_model_metadata(monkeypatch):
    import lmdeploy.turbomind.model_loader as model_loader_mod

    loader = Mock()
    loader.all_items.return_value = {'weight': object()}
    create_loader = Mock(return_value=loader)
    monkeypatch.setattr(model_loader_mod, 'create_loader', create_loader)

    model = Mock()
    model._loader_mappings = []
    model_loader = ModelLoader(model, _FakeModelComm(), 1, 'model-path',
                               data_type=Mock(), engine_config=Mock(attn_tp_size=1, attn_cp_size=1, mlp_tp_size=1))

    assert list(model_loader.export_iter()) == [-1]

    create_loader.assert_called_once_with('model-path', None, [])
    model.set_params.assert_called_once_with(loader.all_items.return_value)
    model.model.assert_called_once_with()
