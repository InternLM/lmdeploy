# Copyright (c) OpenMMLab. All rights reserved.

import json
from pathlib import Path

import pytest
import torch

from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader


def test_model_weight_loader_selects_required_shards(tmp_path: Path, monkeypatch):
    weight_map = {
        'model.layers.0.weight': 'target.safetensors',
        'model.layers.80.weight': 'mtp-a.safetensors',
        'model.layers.80.weight_scale': 'mtp-b.safetensors',
    }
    index = {'metadata': {}, 'weight_map': weight_map}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))

    loader = ModelWeightLoader(str(tmp_path))
    opened_paths = []
    allowed_names_seen = []

    def _get_weights_iterator(path, allowed_names=None):
        opened_paths.append(Path(path).name)
        allowed_names_seen.append(allowed_names)
        return iter(())

    monkeypatch.setattr(loader, '_get_weights_iterator', _get_weights_iterator)

    class _DraftModel(torch.nn.Module):
        @staticmethod
        def get_checkpoint_weight_prefixes():
            return ('model.layers.80.',)

        @staticmethod
        def load_weights(weights):
            list(weights)

    loader.load_model_weights(_DraftModel())

    assert set(opened_paths) == {'mtp-a.safetensors', 'mtp-b.safetensors'}
    assert all(names == {'model.layers.80.weight', 'model.layers.80.weight_scale'} for names in allowed_names_seen)


def test_model_weight_loader_rejects_missing_requested_prefix(tmp_path: Path):
    weight_map = {
        'model.layers.0.weight': 'target.safetensors',
    }
    index = {'metadata': {}, 'weight_map': weight_map}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))

    loader = ModelWeightLoader(str(tmp_path))

    class _DraftModel(torch.nn.Module):
        @staticmethod
        def get_checkpoint_weight_prefixes():
            return ('model.layers.80.',)

        @staticmethod
        def load_weights(weights):
            list(weights)

    with pytest.raises(RuntimeError, match='No checkpoint tensors match'):
        loader.load_model_weights(_DraftModel())


def test_model_weight_loader_preserves_default_full_shard_loading(
    tmp_path: Path,
    monkeypatch,
):
    weight_map = {
        'model.layers.0.weight': 'target-a.safetensors',
        'model.layers.1.weight': 'target-b.safetensors',
    }
    index = {'metadata': {}, 'weight_map': weight_map}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))

    loader = ModelWeightLoader(str(tmp_path))
    opened_paths = []
    allowed_names_seen = []

    def _get_weights_iterator(path, allowed_names=None):
        opened_paths.append(Path(path).name)
        allowed_names_seen.append(allowed_names)
        return iter(())

    monkeypatch.setattr(loader, '_get_weights_iterator', _get_weights_iterator)

    class _RegularModel(torch.nn.Module):
        @staticmethod
        def load_weights(weights):
            list(weights)

    loader.load_model_weights(_RegularModel())

    assert set(opened_paths) == {'target-a.safetensors', 'target-b.safetensors'}
    assert allowed_names_seen == [None, None]
