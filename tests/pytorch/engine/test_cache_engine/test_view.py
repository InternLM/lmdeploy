# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec, LayerRowMap
from lmdeploy.pytorch.engine.cache_engine.view import NamedCacheView


def _tensor_spec(name: str, *, layers=None, consumers=None):
    layer_rows = None if layers is None else LayerRowMap.build(name, layers)
    return CacheTensorSpec(name=name,
                           desc=CacheDesc([1], torch.float32),
                           layer_rows=layer_rows,
                           consumer_rows=consumers)


def test_named_cache_view_keeps_unscoped_mapping_access():
    cache = torch.arange(6).reshape(2, 3)
    view = NamedCacheView((_tensor_spec('plain'), ), (cache, ))

    assert list(view) == ['plain']
    assert len(view) == 1
    assert 'plain' in view
    assert view['plain'] is cache
    with pytest.raises(RuntimeError, match='not bound to consumer rows'):
        view.row('plain', 1)
    with pytest.raises(RuntimeError, match='not bound to model layers'):
        view.layer('plain', 0)


def test_named_cache_view_resolves_consumer_rows_across_tensors():
    tensor_specs = (
        _tensor_spec('index', consumers=(0, 2)),
        _tensor_spec('index', consumers=(1, )),
    )
    caches = (
        torch.tensor([[10], [12]]),
        torch.tensor([[11]]),
    )

    view = NamedCacheView(tensor_specs, caches)

    assert view.row('index', 0).item() == 10
    assert view.row('index', 1).item() == 11
    assert view.row('index', 2).item() == 12
    with pytest.raises(RuntimeError, match='multiple physical tensors'):
        view['index']
    with pytest.raises(RuntimeError, match='Consumer row 3 does not own cache'):
        view.row('index', 3)


def test_named_cache_view_resolves_layer_rows_across_tensors():
    tensor_specs = (
        _tensor_spec('kv', layers=(0, 2)),
        _tensor_spec('kv', layers=(1, )),
    )
    caches = (
        torch.tensor([[20], [22]]),
        torch.tensor([[21]]),
    )

    view = NamedCacheView(tensor_specs, caches)

    assert view.layer('kv', 0).item() == 20
    assert view.layer('kv', 1).item() == 21
    assert view.layer('kv', 2).item() == 22
    with pytest.raises(RuntimeError, match='Layer 3 does not own cache'):
        view.layer('kv', 3)


def test_named_cache_view_requires_one_tensor_per_spec():
    tensor_specs = (_tensor_spec('cache'), )

    with pytest.raises(ValueError, match='same length'):
        NamedCacheView(tensor_specs, ())
