# Copyright (c) OpenMMLab. All rights reserved.
from queue import Queue
from unittest.mock import Mock

from lmdeploy.turbomind.loader import StateDictLoader, create_loader
from lmdeploy.turbomind.model_loader import ModelLoader


class _FakeModelComm:

    def attn_tp_rank(self, gpu): return gpu
    def mlp_tp_rank(self, gpu): return gpu
    def model_tp_rank(self, gpu): return gpu
    def context(self, gpu): return f'ctx:{gpu}'
    def root(self, gpu): return f'root:{gpu}'


def test_state_dict_loader_can_be_created_for_queue():
    queue = Queue()

    loader = create_loader(queue)

    assert isinstance(loader, StateDictLoader)
    assert loader.pattern is None


def test_create_loader_rejects_path_input():
    """Disk-backed paths must use create_checkpoint, not create_loader."""
    import pytest

    with pytest.raises(RuntimeError, match='no longer supports paths'):
        create_loader('/nonexistent')


def test_model_loader_export_uses_checkpoint(monkeypatch):
    """ModelLoader.export builds a Checkpoint and calls model.model(Prefix)."""
    import lmdeploy.turbomind.model_loader as model_loader_mod
    from lmdeploy.turbomind.checkpoint import Prefix

    fake_ckpt = Mock()
    create_checkpoint = Mock(return_value=fake_ckpt)
    monkeypatch.setattr(
        model_loader_mod, 'create_checkpoint', create_checkpoint)

    model = Mock()
    model._loader_mappings = [lambda s: s]

    model_loader = ModelLoader(
        model, _FakeModelComm(), 1, 'model-path',
        data_type=Mock(),
        engine_config=Mock(attn_tp_size=1, attn_cp_size=1, mlp_tp_size=1))

    model_loader.export()

    create_checkpoint.assert_called_once_with(
        'model-path', mappings=model._loader_mappings)
    args, kwargs = model.model.call_args
    assert kwargs == {}
    pfx, = args
    assert isinstance(pfx, Prefix)
    assert pfx.ckpt is fake_ckpt
    assert pfx.prefix == ''
    fake_ckpt.close.assert_called_once_with()
