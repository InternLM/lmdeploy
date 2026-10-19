# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in compact layout foundation; not runtime/Store integration tests."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CompactStateCapacity, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine import state as state_module
from lmdeploy.pytorch.engine.cache_engine.layout import ContiguousStateCacheLayout, PackedStateCacheLayout
from lmdeploy.pytorch.engine.cache_engine.state import CompactStatePlan
from lmdeploy.pytorch.engine.cache_engine.view import NamedCacheView


@pytest.mark.parametrize('w', [1, 2, 4, 6])
@pytest.mark.parametrize('checkpoint_budget,store_budget', [(0, 0), (1, 0), (32, 4)])
def test_capacity_preserves_legacy_request_admission(w, checkpoint_budget, store_budget):
    capacity = CompactStateCapacity.from_legacy(18 + checkpoint_budget + store_budget,
                                                checkpoint_budget, store_budget, w)
    assert capacity.request_slots == 18
    assert capacity.request_capacity == 17
    assert capacity.bank_width == w + 1
    assert capacity.num_units == 1 + 17 * (w + 1) + checkpoint_budget + store_budget


def test_stateless_model_has_no_compact_allocation(monkeypatch):
    capacity = CompactStateCapacity.from_legacy(None, 32, 4, 6)
    assert (capacity.request_slots, capacity.bank_width, capacity.num_units) == (0, 0, 0)
    backend = SimpleNamespace(build_state_layout=lambda _: pytest.fail('Stateless layout must not be built'))
    monkeypatch.setattr(state_module, 'get_backend', lambda: SimpleNamespace(get_cache_backend=lambda: backend))
    plan = CompactStatePlan([], capacity)
    assert plan.get_nbytes() == (0, 0)
    assert all(not allocation.pools and not allocation.tensor_views for allocation in plan.allocate('cpu'))


@pytest.mark.parametrize('args', [(-1, 1), (1, 0), (0, 1), (0, 0, 1), (2, 1, -1), (2, 1, 0, -1), (2, 1.5)])
def test_invalid_capacity(args):
    with pytest.raises(ValueError):
        CompactStateCapacity(*args)


def test_invalid_legacy_capacity():
    with pytest.raises(ValueError, match='reserved request slot'):
        CompactStateCapacity.from_legacy(5, 4, 1, 1)


@pytest.mark.parametrize('layout_type', [PackedStateCacheLayout, ContiguousStateCacheLayout])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mixed_dtype_layer_rows_and_auxiliary_owning_bytes(monkeypatch, layout_type, device):
    builds = []

    def build(specs):
        builds.append(specs)
        return layout_type(tuple(specs))

    backend = SimpleNamespace(build_state_layout=build)
    monkeypatch.setattr(state_module, 'get_backend', lambda: SimpleNamespace(get_cache_backend=lambda: backend))
    specs = [
        StateCacheSpec('recurrent', (2, 4, 4), torch.float32, layer_ids=[1, 9]),
        StateCacheSpec('conv', (12, 3), torch.bfloat16, layer_ids=[1, 9]),
        StateCacheSpec('tail', (7, 4), torch.float16, layer_ids=[0, 3],
                       prefix_checkpoint=False, store_cacheable=False),
    ]
    capacity = CompactStateCapacity.from_legacy(9, 2, 1, 4)
    plan = CompactStatePlan(specs, capacity)
    expected = plan.get_nbytes()
    units, auxiliary = plan.allocate(device)
    assert len(builds) == 2  # allocation and sizing reuse the compiled layouts
    assert (units.nbytes, auxiliary.nbytes) == expected
    assert units.nbytes == sum(p.tensor.untyped_storage().nbytes() for p in units.pools)
    assert auxiliary.nbytes == sum(p.tensor.untyped_storage().nbytes() for p in auxiliary.pools)
    assert units.tensor_views[0].shape == (2, capacity.num_units, 2, 4, 4)
    assert auxiliary.tensor_views[0].shape == (2, capacity.request_slots, 7, 4)
    assert {p.tensor.untyped_storage().data_ptr() for p in units.pools}.isdisjoint(
        p.tensor.untyped_storage().data_ptr() for p in auxiliary.pools)
    assert all(p.transfer_ranges == () for p in auxiliary.pools)
    view = NamedCacheView(plan.unit_specs, units.tensor_views)
    view.layer('conv', 9)[3].fill_(7)
    assert torch.all(units.tensor_views[1][1, 3] == 7)
    assert torch.count_nonzero(units.tensor_views[1][0]) == 0
    # Auxiliary cost is per request, independent of W and checkpoint credits.
    bigger = CompactStatePlan(specs, CompactStateCapacity.from_legacy(14, 7, 1, 6))
    assert bigger.get_nbytes()[1] == expected[1]
    assert bigger.get_nbytes()[0] > expected[0]


@pytest.mark.parametrize('local,store', [(True, False), (False, True)])
def test_local_and_store_field_mismatch_rejected(local, store):
    spec = StateCacheSpec('state', (4,), torch.float32, prefix_checkpoint=local, store_cacheable=store)
    with pytest.raises(ValueError, match='matching local and Store'):
        CompactStatePlan([spec], CompactStateCapacity(2, 1))


def test_declarations_and_capacity_must_agree():
    with pytest.raises(ValueError, match='must agree'):
        CompactStatePlan([], CompactStateCapacity(2, 1))
    specs = [StateCacheSpec('state', (4,), torch.float32)]
    with pytest.raises(ValueError, match='must agree'):
        CompactStatePlan(specs, CompactStateCapacity(0, 0))
    with pytest.raises(ValueError, match='unique names'):
        CompactStatePlan(specs * 2, CompactStateCapacity(2, 1))
