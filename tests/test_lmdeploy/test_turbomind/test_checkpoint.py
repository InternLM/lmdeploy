# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for Checkpoint subclasses and create_checkpoint factory."""
from __future__ import annotations

import json
import os.path as osp

import pytest
import torch
from safetensors.torch import save_file

from lmdeploy.turbomind.checkpoint import (
    SAFE_WEIGHT_INDEX_NAME,
    WEIGHT_INDEX_NAME,
    Checkpoint,
    PytorchCheckpoint,
    SafetensorsCheckpoint,
    create_checkpoint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_safetensors_shards(root: str, weight_map: dict[str, str],
                              tensors_per_shard: dict[str, dict[str, torch.Tensor]],
                              with_index: bool = True) -> None:
    """Write 1+ safetensors shards under ``root``.

    ``weight_map`` maps tensor name -> shard filename. If ``with_index``
    is True, also writes ``model.safetensors.index.json``.
    """
    for shard_name, tensors in tensors_per_shard.items():
        save_file(tensors, osp.join(root, shard_name))
    if with_index:
        with open(osp.join(root, 'model.safetensors.index.json'), 'w') as f:
            json.dump({'weight_map': weight_map}, f)


def _write_pytorch_shards(root: str, weight_map: dict[str, str],
                          tensors_per_shard: dict[str, dict[str, torch.Tensor]]) -> None:
    for shard_name, tensors in tensors_per_shard.items():
        torch.save(tensors, osp.join(root, shard_name))
    with open(osp.join(root, 'pytorch_model.bin.index.json'), 'w') as f:
        json.dump({'weight_map': weight_map}, f)


# ---------------------------------------------------------------------------
# SafetensorsCheckpoint
# ---------------------------------------------------------------------------


class TestSafetensorsCheckpoint:

    def test_get_returns_tensor(self, tmp_path):
        a = torch.arange(4, dtype=torch.float32)
        b = torch.arange(8, dtype=torch.float32)
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors', 'b': 'model-00002.safetensors'},
            tensors_per_shard={
                'model-00001.safetensors': {'a': a},
                'model-00002.safetensors': {'b': b},
            },
        )
        ckpt = SafetensorsCheckpoint(str(tmp_path),
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        torch.testing.assert_close(ckpt.get('a').cpu(), a)
        torch.testing.assert_close(ckpt.get('b').cpu(), b)

    def test_get_missing_raises_key_error(self, tmp_path):
        a = torch.zeros(2)
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors': {'a': a}},
        )
        ckpt = SafetensorsCheckpoint(str(tmp_path),
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        with pytest.raises(KeyError):
            ckpt.get('missing')

    def test_has_reflects_keys(self, tmp_path):
        a = torch.zeros(2)
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors': {'a': a}},
        )
        ckpt = SafetensorsCheckpoint(str(tmp_path),
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        assert ckpt.has('a') is True
        assert ckpt.has('missing') is False

    def test_keys_returns_all_keys(self, tmp_path):
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors',
                        'b': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors':
                               {'a': torch.zeros(2), 'b': torch.zeros(2)}},
        )
        ckpt = SafetensorsCheckpoint(str(tmp_path),
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        assert set(ckpt.keys()) == {'a', 'b'}

    def test_mappings_rewrite_keys_on_load(self, tmp_path):
        """Per-model regex remappings are applied during construction."""
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'experts.gate_up_proj': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors':
                               {'experts.gate_up_proj': torch.zeros(2)}},
        )
        import re

        def add_weight_suffix(s: str) -> str:
            return re.sub(r'(experts\.[a-z_]+_proj)$', r'\1.weight', s)

        ckpt = SafetensorsCheckpoint(str(tmp_path), mappings=[add_weight_suffix],
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        assert ckpt.has('experts.gate_up_proj.weight')
        assert not ckpt.has('experts.gate_up_proj')


# ---------------------------------------------------------------------------
# PytorchCheckpoint
# ---------------------------------------------------------------------------


class TestPytorchCheckpoint:

    def test_get_returns_tensor(self, tmp_path):
        a = torch.arange(4, dtype=torch.float32)
        _write_pytorch_shards(
            str(tmp_path),
            weight_map={'a': 'pytorch_model-00001-of-00001.bin'},
            tensors_per_shard={'pytorch_model-00001-of-00001.bin': {'a': a}},
        )
        ckpt = PytorchCheckpoint(str(tmp_path), index_name=WEIGHT_INDEX_NAME)
        torch.testing.assert_close(ckpt.get('a').cpu(), a)


# ---------------------------------------------------------------------------
# create_checkpoint factory dispatch
# ---------------------------------------------------------------------------


class TestCreateCheckpoint:

    def test_safetensors_dir_returns_safetensors_checkpoint(self, tmp_path):
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors': {'a': torch.zeros(2)}},
        )
        assert isinstance(create_checkpoint(str(tmp_path)), SafetensorsCheckpoint)

    def test_pytorch_dir_returns_pytorch_checkpoint(self, tmp_path):
        _write_pytorch_shards(
            str(tmp_path),
            weight_map={'a': 'pytorch_model-00001-of-00001.bin'},
            tensors_per_shard={'pytorch_model-00001-of-00001.bin': {'a': torch.zeros(2)}},
        )
        assert isinstance(create_checkpoint(str(tmp_path)), PytorchCheckpoint)

    def test_no_checkpoint_files_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match='Failed to find valid'):
            create_checkpoint(str(tmp_path))


# ---------------------------------------------------------------------------
# Close()
# ---------------------------------------------------------------------------


class TestCheckpointClose:

    def test_close_is_idempotent(self, tmp_path):
        _write_safetensors_shards(
            str(tmp_path),
            weight_map={'a': 'model-00001.safetensors'},
            tensors_per_shard={'model-00001.safetensors': {'a': torch.zeros(2)}},
        )
        ckpt = SafetensorsCheckpoint(str(tmp_path),
                                     index_name=SAFE_WEIGHT_INDEX_NAME)
        ckpt.close()
        ckpt.close()  # second call is a no-op

    def test_base_class_close_default_is_noop(self):
        class _Stub(Checkpoint):
            def get(self, key, index=None): return None
            def has(self, key): return False
            def pop(self, key, index=None): return None
            def keys(self): return iter(())
        _Stub().close()  # does not raise
