# Copyright (c) OpenMMLab. All rights reserved.
"""Exclusive serving capacity, lifecycle and actual worker allocations."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.compact_state import CompactStateInputs
from lmdeploy.pytorch.config import CacheConfig, CompactStateCapacity, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.state import CompactStatePlan, StateCacheEngine
from lmdeploy.pytorch.paging.compact_state import CompactStateManager
from lmdeploy.pytorch.strategies.base.model_inputs import MakeDummyMeta, make_dummy_inputs
from tests.pytorch.compact_state_reference import StateAccessRow, freeze_ready, prepare_prefill_output, submit


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('checkpoint_budget,store_budget', [(0, 0), (1, 0), (1, 2)])
def test_exclusive_credits_promotion_and_deferred_recycling(width, checkpoint_budget, store_budget):
    capacity = CompactStateCapacity(3, width, checkpoint_budget, store_budget, allow_input_alias=True)
    owner = CompactStateManager(capacity)
    banks = [owner.allocate_bank(prefill=True) for _ in range(2)]
    initial = [b.unit_ids for b in banks]
    assert capacity.num_units == 1 + 2 * width + checkpoint_budget + store_budget
    assert owner.get_num_free_optional() == checkpoint_budget + store_budget
    first = owner.submit_prefill(tuple(banks))
    for bank in banks:
        source = bank.unit_ids[0]
        assert prepare_prefill_output(owner, bank, source) == source
        owner.prepare_prefill_bank(bank, private_input=True)
        assert len(bank.unit_ids) == 1
    second = submit(owner, tuple(StateAccessRow(b, b.unit_ids[0], b.unit_ids) for b in banks))
    for bank in banks:
        assert owner.promote_to_decode(bank)
        assert len(bank.unit_ids) == width
    decode = owner.submit_decode(tuple(banks))
    assert [tuple(ids) for ids in first.bank_ids] == initial
    for bank in banks:
        owner.release_bank(bank)
    owner.complete(first)
    owner.complete(second)
    assert len(owner._banks) == 2
    owner.complete(decode)
    assert not owner._units and not owner._banks and not owner._accesses
    assert owner.get_num_free() == capacity.num_units - 1


def test_exclusive_alias_rejects_early_shared_and_other_request_writes():
    owner = CompactStateManager(CompactStateCapacity(3, 4, 1, allow_input_alias=True))
    a, b = [owner.allocate_bank() for _ in range(2)]
    source, next_id = a.unit_ids[:2]
    for outputs in [(source, next_id), (source, source), (b.unit_ids[0], source)]:
        with pytest.raises(ValueError):
            submit(owner, (StateAccessRow(a, source, outputs),))
        assert not owner._accesses
    ok = submit(owner, (StateAccessRow(a, source, (next_id, source)),))
    owner.complete(ok)
    frozen = freeze_ready(owner, a, source, 64)
    owner.bind_input(a, frozen)
    owner.bind_input(b, frozen)
    for bank in (a, b):
        with pytest.raises(ValueError, match='Output must be a private'):
            submit(owner, (StateAccessRow(bank, source, (source,)),))
    read = submit(owner, tuple(StateAccessRow(bank, source, (bank.unit_ids[0],)) for bank in (a, b)))
    owner.release_reference(frozen)
    for bank in (a, b):
        owner.release_bank(bank)
    assert source in owner._units
    owner.complete(read)
    assert not owner._units


def test_single_unit_ar_freeze_replaces_writer_and_retains_shared_input():
    owner = CompactStateManager(CompactStateCapacity(2, 1, 1, allow_input_alias=True))
    bank = owner.allocate_bank()
    old = bank.unit_ids[0]
    frozen = freeze_ready(owner, bank, old, 64)
    owner.bind_input(bank, frozen)
    assert len(bank.unit_ids) == 1 and old not in bank.unit_ids
    read = submit(owner, (StateAccessRow(bank, old, bank.unit_ids),))
    owner.release_reference(frozen)
    owner.release_bank(bank)
    assert old in owner._units
    owner.complete(read)
    assert not owner._units


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('auxiliary', [False, True])
def test_real_worker_owning_storage_shrinks_without_request_or_kv_cut(width, auxiliary):
    # A=10 => R=9, B=1: AR shrinks the real allocation from 20 to 11 units.
    old = CompactStateCapacity(10, width, 1)
    new = replace(old, allow_input_alias=True)
    assert old.request_capacity == new.request_capacity == 9
    specs = [StateCacheSpec('recurrent', (width, 2, 16, 16), torch.bfloat16,
                            layer_ids=[0, 2], compact_shape=(2, 16, 16)),
             StateCacheSpec('conv', (64, width + 3), torch.bfloat16,
                            layer_ids=[0, 2], compact_shape=(64, 3))]
    if auxiliary:
        specs.append(StateCacheSpec('tail', (8, 16), torch.float32, layer_ids=[1],
                                    prefix_checkpoint=False, store_cacheable=False))
    model = SimpleNamespace(state_cache_specs=specs, states_shapes=[])
    cache = CacheConfig(max_batches=9, block_size=64, num_cpu_blocks=0, num_gpu_blocks=512,
                        num_state_caches=11)
    engines = []
    for capacity in (old, new):
        plan = CompactStatePlan.from_model_config(model, capacity)
        engines.append(StateCacheEngine(cache, model, compact_plan=plan))
        units = engines[-1].unit_allocation
        assert units.nbytes == sum(p.tensor.untyped_storage().nbytes() for p in units.pools)
        assert units.nbytes == plan.get_nbytes()[0]
        assert all(v.shape[1] == capacity.num_units for v in units.tensor_views)
    before, after = engines
    per_unit = before.unit_allocation.nbytes // old.num_units
    assert before.unit_allocation.nbytes - after.unit_allocation.nbytes == 9 * per_unit
    assert before.auxiliary_allocation.nbytes == after.auxiliary_allocation.nbytes
    assert before.cache_config is after.cache_config
    assert cache.num_gpu_blocks == 512 and cache.max_batches == 9
    if width == 1:
        assert (old.num_units, new.num_units) == (20, 11)


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('decode', [False, True])
def test_exclusive_dummy_and_metadata_preserve_permission(width, decode):
    dummy = make_dummy_inputs(3, width if decode else 72, decode, device='cuda',
                              meta=MakeDummyMeta(use_ssm=True, compact_bank_width=width,
                                                 compact_allow_input_alias=True))
    compact = dummy.compact_state
    assert compact.allow_input_alias and compact.bank_ids.shape == (3, width)
    selected = compact.index_select(torch.tensor([2, 0], device='cuda')).to_device('cpu')
    merged = selected.merge(selected)
    assert merged.allow_input_alias and merged.bank_ids.shape == (4, width)
    assert (merged.bank_ids == -1).all()
    if width > 1:
        disjoint = CompactStateInputs(compact.bank_ids, compact.input_ids, compact.output_ids[:, :1].contiguous())
        with pytest.raises(ValueError, match='Merging'):
            compact.merge(disjoint)
