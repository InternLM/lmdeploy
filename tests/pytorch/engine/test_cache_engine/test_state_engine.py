# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.state as state_module
from lmdeploy.pytorch.config import CacheConfig, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import (
    CacheAllocation,
    CachePool,
    ContiguousStateCacheLayout,
    PackedStateCacheLayout,
)
from lmdeploy.pytorch.engine.cache_engine.schema import build_state_cache_tensor_specs


def _allocate_test_state_caches(num_caches: int,
                                state_shapes,
                                device: torch.device | str = 'cpu',
                                state_specs=None) -> CacheAllocation:
    tensor_specs = build_state_cache_tensor_specs(state_shapes, state_specs=state_specs)
    return state_module._allocate_state_caches(tensor_specs, num_caches=num_caches, device=device)


def test_layered_state_cache_specs_do_not_require_total_layer_count():
    state_specs = [StateCacheSpec('subset', (96, ), torch.float32, layer_ids=[1, 9])]
    state_shapes = [(spec.shape, spec.dtype) for spec in state_specs]

    allocation = _allocate_test_state_caches(num_caches=2,
                                             state_shapes=state_shapes,
                                             state_specs=state_specs)
    mem_pool = allocation.pools[0].tensor
    caches = allocation.tensor_views

    assert isinstance(allocation, CacheAllocation)
    assert tuple(mem_pool.shape) == (2, 768)
    assert tuple(caches[0].shape) == (2, 2, 96)
    assert StateCacheEngine.get_state_slot_nbytes(state_shapes, state_specs=state_specs) == 768


def _make_multi_pool_state_allocation(num_caches: int, device: torch.device | str = 'cpu'):
    first = torch.zeros((num_caches, 2), dtype=torch.float32, device=device)
    second = torch.zeros((3, num_caches, 2), dtype=torch.float16, device=device)
    return CacheAllocation(
        pools=(CachePool(first, entry_axis=0), CachePool(second, entry_axis=1)),
        tensor_views=(first, second),
    )


def test_state_cache_engine_accepts_multi_pool_layout(monkeypatch):
    layout = SimpleNamespace(
        allocate=lambda num_caches, device: _make_multi_pool_state_allocation(num_caches, device),
    )
    cache_backend = SimpleNamespace(build_state_layout=lambda tensor_specs: layout)
    ops_backend = SimpleNamespace(get_cache_backend=lambda: cache_backend)
    monkeypatch.setattr(state_module, 'get_backend', lambda: ops_backend)

    allocation = _allocate_test_state_caches(num_caches=2, state_shapes=[])

    assert [pool.entry_axis for pool in allocation.pools] == [0, 1]
    assert allocation.nbytes == 2 * (2 * 4 + 3 * 2 * 2)
    assert StateCacheEngine.get_state_slot_nbytes([]) == 2 * 4 + 3 * 2 * 2


def test_layer_scoped_state_cache_specs_reject_invalid_layer_ids():
    negative_layer = [StateCacheSpec('bad', (1, ), torch.float32, layer_ids=[-1])]
    with pytest.raises(ValueError, match='non-negative'):
        _allocate_test_state_caches(num_caches=1,
                                    state_shapes=[((1, ), torch.float32)],
                                    state_specs=negative_layer,
                                    device='meta')

    empty_state_layers = [StateCacheSpec('empty', (1, ), torch.float32, layer_ids=[])]
    with pytest.raises(ValueError, match='must not be empty'):
        _allocate_test_state_caches(num_caches=1,
                                    state_shapes=[((1, ), torch.float32)],
                                    state_specs=empty_state_layers,
                                    device='meta')


def _make_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.compact_plan = None
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[((2, 3), torch.float32), ((2, ), torch.float16)])
    cache_engine.allocation = _allocate_test_state_caches(num_caches=num_caches,
                                                          state_shapes=cache_engine.cache_config.states_shapes)
    cache_engine._cache_tensors = list(cache_engine.allocation.tensor_views)
    cache_engine._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in cache_engine.allocation.pools)
    return cache_engine


def _make_multi_pool_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.compact_plan = None
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[])
    cache_engine.allocation = _make_multi_pool_state_allocation(num_caches)
    cache_engine._cache_tensors = list(cache_engine.allocation.tensor_views)
    cache_engine._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in cache_engine.allocation.pools)
    return cache_engine


def test_connector_state_cache_pools_expose_owning_pools_and_axes():
    cache_engine = _make_multi_pool_state_cache_engine()

    pools = cache_engine.connector_state_cache_pools

    assert [pool.entry_axis for pool in pools] == [0, 1]
    assert all(pool.tensor is allocation_pool.tensor
               for pool, allocation_pool in zip(pools, cache_engine.allocation.pools))


@pytest.fixture(params=['cpu'])
def glm_state_cache_engine(monkeypatch, request):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    monkeypatch.setattr(state_module, '_allocate_state_caches',
                        lambda specs, num_caches, device: PackedStateCacheLayout(tuple(specs)).allocate(
                            num_caches, device=request.param))
    hf_config = Glm5NextConfig(text_config=dict(
        num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
        linear_num_heads=2, linear_head_dim=8, index_head_dim=8))
    # This fixture exercises the legacy ring ABI, including on a CUDA test device.
    model_config = Glm5NextModelConfigBuilder.build(hf_config, num_spec_tokens=3, device_type='cpu')
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0,
                               num_gpu_blocks=0, num_state_caches=4, states_shapes=model_config.states_shapes)
    engine = StateCacheEngine(cache_config, model_config)
    return engine, model_config


def test_glm_store_projection_preserves_local_state_copies(glm_state_cache_engine):
    engine, model_config = glm_state_cache_engine
    assert [spec.store_cacheable for spec in model_config.state_cache_specs] == [True, True, False, False]
    checkpoint_pool, runtime_pool = engine.connector_state_cache_pools
    assert checkpoint_pool.transfer_ranges is None
    assert runtime_pool.transfer_ranges == ()
    assert checkpoint_pool.tensor is engine.allocation.pools[0].tensor
    assert runtime_pool.tensor is engine.allocation.pools[1].tensor
    for value, cache in enumerate(engine.state_caches, start=1):
        cache[1].fill_(value)
    engine.copy_slots(1, 3)
    for value, cache in enumerate(engine.state_caches, start=1):
        assert (cache[3] == value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('glm_state_cache_engine', ['cuda'], indirect=True)
@pytest.mark.parametrize('boundary_offset', [1, 2, 4])
def test_glm_checkpoint_restores_the_selected_verification_position(
        monkeypatch, glm_state_cache_engine, boundary_offset):
    from lmdeploy.pytorch.backends import gated_delta_rule as meta_module

    engine, _ = glm_state_cache_engine
    conv, recurrent, tail_keys, tail_scores = engine.state_caches
    for cache in engine.state_caches:
        cache.fill_(-10)
    # Mark every entry by its absolute position, including the unaccepted tail.
    boundary, num_spec_tokens = 256, 3
    history = boundary - boundary_offset
    ring_size = num_spec_tokens + 1
    conv_length = conv.shape[-1]
    conv_width = conv_length - num_spec_tokens
    for position in range(history + 1, history + ring_size + 1):
        recurrent[1, 0, position % ring_size].fill_(position)
    for position in range(history + ring_size - conv_length, history + ring_size):
        conv[1, 0, :, position % conv_length].fill_(position)
    plan = torch.tensor([[0, 0], [boundary, boundary], [1, 1], [2, 3]], device='cuda')
    engine.copy_decode_slots(plan, torch.tensor([True, False], device='cuda'))
    assert (recurrent[3] == -10).all()
    for cache in engine.state_caches:
        cache[1].fill_(999)  # Later forwards reuse the runtime state.
    engine.copy_slots(2, 3, checkpoint_only=True)

    # Use the actual prefill restore metadata to select the saved boundary.
    monkeypatch.setattr(meta_module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=num_spec_tokens)))
    meta = meta_module.GatedDeltaMeta(1, conv_width, torch.tensor([3]), SimpleNamespace(
        is_decoding=False, cu_seqlens_q=torch.tensor([0, 1]),
        q_seqlens=torch.tensor([1]), kv_seqlens=torch.tensor([boundary + 1])))
    restored_state = recurrent[3, 0, meta.spec_state_offsets[0][0]]
    torch.testing.assert_close(restored_state, torch.full_like(restored_state, boundary))
    restored_conv = conv[3, 0, :, meta.spec_conv_offsets[0][0]]
    expected_conv = torch.arange(boundary - conv_width + 1, boundary, dtype=conv.dtype, device=conv.device)
    torch.testing.assert_close(restored_conv, expected_conv.expand_as(restored_conv))
    assert (tail_keys[3] == -10).all() and (tail_scores[3] == -10).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('boundary_offset', [1, 2, 4])
def test_glm_compact_checkpoint_copies_exact_accepted_state_unit(monkeypatch, boundary_offset):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    model = Glm5NextModelConfigBuilder.build(Glm5NextConfig(text_config=dict(
        num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
        linear_num_heads=2, linear_head_dim=8, index_head_dim=8)), num_spec_tokens=3)
    cache = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0,
                        num_state_caches=4, states_shapes=model.states_shapes, prefix_cache_token_lookahead=1)
    plan = state_module.SSMStatePlan.from_model_config(model, SSMStateCapacity(2, 4, 2))
    engine = StateCacheEngine(cache, model, compact_plan=plan)
    conv, recurrent, tail_keys, tail_scores = engine.state_caches
    for tensor in engine.state_caches:
        tensor.fill_(-10)
    boundary = 256
    history = boundary - boundary_offset
    outputs = torch.tensor([[3, 1, 4, 2]], device='cuda')
    for offset, unit in enumerate(outputs[0].tolist(), start=1):
        recurrent[unit].fill_(history + offset)
        conv[unit].copy_(torch.arange(history + offset - conv.shape[-1], history + offset,
                                     device='cuda', dtype=conv.dtype).expand_as(conv[unit]))
    agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
    agent.need_output, agent.cache_config, agent.state_cache_engine = True, cache, engine
    monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: SimpleNamespace(
        current_context=lambda: SimpleNamespace(dist_config=SimpleNamespace(attn_tp=1))))
    inputs = SimpleNamespace(history_lengths=torch.tensor([history], device='cuda'),
                             seq_length=torch.tensor([4], device='cuda'), is_decoding=True,
                             ssm_state_meta=SimpleNamespace(output_state_ids=outputs))
    output = agent_module.BatchedOutputs(
        next_token_ids=torch.tensor([[7] * boundary_offset + [-1] * (4 - boundary_offset)], device='cuda'),
        stopped=torch.tensor([False], device='cuda'), stop_pos=torch.tensor([-1], device='cuda'))
    copies = torch.tensor([[0, 0], [boundary, boundary + 4], [-1, -1], [5, 6]], device='cuda')
    agent._save_decode_checkpoints(CacheCheckpointInputs(decode_save_plan=copies), inputs, output)
    assert output.decode_checkpoint_mask.tolist() == [True, False]
    conv[1:5].fill_(999)
    recurrent[1:5].fill_(999)
    assert (recurrent[5] == boundary).all() and (recurrent[6] == -10).all()
    expected = torch.arange(boundary - conv.shape[-1], boundary, device='cuda', dtype=conv.dtype)
    torch.testing.assert_close(conv[5], expected.expand_as(conv[5]))
    assert (conv[6] == -10).all() and (tail_keys == -10).all() and (tail_scores == -10).all()


def test_state_cache_engine_zero_slots_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first.fill_(1)
    second.fill_(1)

    cache_engine.zero_slots(torch.tensor([1, 3]), torch.tensor([True, False]))

    assert torch.count_nonzero(first[1]) == 0
    assert torch.count_nonzero(second[:, 1]) == 0
    assert torch.all(first[3] == 1)
    assert torch.all(second[:, 3] == 1)


def test_state_cache_engine_copy_slots_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first[0].fill_(3)
    first[1].fill_(5)
    second[:, 0].fill_(7)
    second[:, 1].fill_(9)

    cache_engine.copy_slots((1, 0), (3, 2))

    assert torch.equal(first[2], first[0])
    assert torch.equal(first[3], first[1])
    assert torch.equal(second[:, 2], second[:, 0])
    assert torch.equal(second[:, 3], second[:, 1])


def test_state_cache_engine_copy_slots_copies_all_state_views():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(5.0)

    cache_engine.copy_slots(1, 2)

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_slots_supports_batched_indices():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[0].fill_(1.0)
    recurrent_state[0].fill_(2.0)
    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(4.0)

    cache_engine.copy_slots((1, 0), (3, 2))

    assert torch.equal(conv_state[2], conv_state[0])
    assert torch.equal(recurrent_state[2], recurrent_state[0])
    assert torch.equal(conv_state[3], conv_state[1])
    assert torch.equal(recurrent_state[3], recurrent_state[1])


def test_state_cache_engine_copy_slots_accepts_host_integer_scalars():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(7.0)
    recurrent_state[1].fill_(9.0)

    cache_engine.copy_slots(np.int64(1), np.int64(2))

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_slots_coalesces_contiguous_ranges():
    ranges = list(StateCacheEngine._coalesce_copy_ranges([4, 1, 5, 0, 6, 9], [20, 11, 21, 10, 22, 30]))

    assert ranges == [(0, 10, 2), (4, 20, 3), (9, 30, 1)]
    assert list(StateCacheEngine._coalesce_copy_ranges([], [])) == []


def test_state_cache_engine_copy_slots_rejects_mismatched_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='same number of elements'):
        cache_engine.copy_slots([0, 1], [2])


def test_state_cache_engine_copy_slots_rejects_tensor_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_slots(torch.tensor([0, 1]), torch.tensor([2, 3]))


def test_state_cache_engine_copy_slots_rejects_tensor_sequence_items():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_slots([torch.tensor(0)], [2])


def test_state_cache_engine_copy_slots_rejects_out_of_range_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_slots([-1], [2])

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_slots([0], [4])


def test_state_cache_engine_copy_slots_rejects_overlapping_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='must not overlap'):
        cache_engine.copy_slots([0, 1], [1, 2])


def test_state_cache_engine_copy_slots_rejects_duplicate_destinations():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='duplicate'):
        cache_engine.copy_slots([0, 1], [2, 2])


def _make_v4_transaction_engine(num_layers=2, num_slots=3):
    engine = object.__new__(StateCacheEngine)
    engine._named_state_caches = {
        'v4_compress_state_r4': torch.arange(
            num_layers * num_slots * 16).reshape(
                num_layers, num_slots, 16, 1).float(),
        'v4_compress_state_r4_idx': torch.arange(
            num_layers * num_slots * 16).reshape(
                num_layers, num_slots, 16, 1).float() + 1000,
        'v4_compress_state_r128': torch.arange(
            num_layers * num_slots * 256).reshape(
                num_layers, num_slots, 256, 1).float(),
    }
    return engine


def test_v4_speculative_state_transaction_restores_only_rejected_rows():
    engine = _make_v4_transaction_engine()
    state_offsets = torch.tensor([1, 2])
    before = {
        name: cache.clone()
        for name, cache in engine.named_state_caches.items()
    }
    transaction = engine.begin_v4_speculative_transaction(
        state_offsets,
        start_positions=torch.tensor([6, 126]),
        q_seqlens=torch.tensor([6, 6]),
        max_q_seqlen=6,
    )

    for name, cache in engine.named_state_caches.items():
        rows, _ = transaction['snapshots'][name]
        for slot_index, slot in enumerate(state_offsets):
            cache[:, slot, rows[slot_index]] = -1

    engine.finish_v4_speculative_transaction(
        transaction, num_rejected_tokens=torch.tensor([4, 1]))

    accepted = (2, 5)
    for name, cache in engine.named_state_caches.items():
        rows, (snapshot, ) = transaction['snapshots'][name]
        width = transaction['max_q_seqlen']
        for slot_index, slot in enumerate(state_offsets):
            accepted_mask = torch.arange(width) < accepted[slot_index]
            if rows.size(1) == 2 * width:
                accepted_mask = torch.cat([accepted_mask, accepted_mask])
            actual = cache[:, slot, rows[slot_index]]
            expected = torch.where(
                accepted_mask.view(1, -1, 1),
                torch.full_like(snapshot[:, slot_index], -1),
                snapshot[:, slot_index])
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(cache[:, 0], before[name][:, 0])

    assert transaction['snapshots']['v4_compress_state_r4'][1][0].size(2) == 12


def test_v4_speculative_state_transaction_rejects_ring_aliasing():
    engine = object.__new__(StateCacheEngine)
    engine._named_state_caches = {
        'v4_compress_state_r4': torch.zeros(1, 1, 16, 1),
    }

    with pytest.raises(RuntimeError, match='query length 9, capacity 8'):
        engine.begin_v4_speculative_transaction(
            state_offsets=torch.tensor([0]),
            start_positions=torch.tensor([0]),
            q_seqlens=torch.tensor([9]),
            max_q_seqlen=9,
        )


@pytest.mark.parametrize('accepted', range(7))
def test_v4_speculative_state_transaction_all_accept_lengths(accepted):
    """Every accepted prefix commits and every rejected suffix rolls back."""
    engine = _make_v4_transaction_engine(num_layers=1, num_slots=2)
    transaction = engine.begin_v4_speculative_transaction(
        state_offsets=torch.tensor([1]),
        start_positions=torch.tensor([127]),
        q_seqlens=torch.tensor([6]),
        max_q_seqlen=6,
    )

    for name, cache in engine.named_state_caches.items():
        rows, _ = transaction['snapshots'][name]
        cache[:, torch.tensor([[1]]), rows] = -1

    engine.finish_v4_speculative_transaction(
        transaction, num_rejected_tokens=torch.tensor([6 - accepted]))

    for name, cache in engine.named_state_caches.items():
        rows, (before, ) = transaction['snapshots'][name]
        actual = cache[:, torch.tensor([[1]]), rows]
        accepted_mask = torch.arange(6) < accepted
        if rows.size(1) == 12:
            accepted_mask = torch.cat([accepted_mask, accepted_mask])
        expected = torch.where(
            accepted_mask.view(1, 1, -1, 1),
            torch.full_like(before, -1), before)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('inactive', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_v4_transaction_ignores_padding_and_keeps_rejection_row_mapping(inactive, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    engine = _make_v4_transaction_engine()
    engine._named_state_caches = {name: cache.to(device) for name, cache in engine.named_state_caches.items()}
    slots = torch.tensor([-1, -1, -1] if inactive else [1, -1, 2], device=device)
    starts = torch.tensor([6, 126, 127], device=device)
    lengths = torch.tensor([6, 3, 4], device=device)
    rejected = torch.tensor([4, 3, 1], device=device)
    before = {name: cache.clone() for name, cache in engine.named_state_caches.items()}
    transaction = engine.begin_v4_speculative_transaction(slots, starts, lengths, 6)
    # Advance even the final slot: padding must not roll back this unrelated
    # change, nor overwrite the accepted prefix of the live row using slot 2.
    for cache in engine.named_state_caches.values():
        cache.add_(100)
    engine.finish_v4_speculative_transaction(transaction, rejected)
    for name, cache in engine.named_state_caches.items():
        expected = before[name] + 100
        for row, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            positions = starts[row] + torch.arange(lengths[row] - rejected[row], lengths[row], device=device)
            capacity = cache.size(2) // 2
            rows = (positions + (4 if '_r4' in name else 0)) % capacity
            rows = torch.cat([rows, rows + capacity])
            expected[:, slot, rows] = before[name][:, slot, rows]
        torch.testing.assert_close(cache, expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_v4_window_transaction_restores_physical_payload_and_scales(device):
    if device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip('Real FP8 pack kernel requires SM90+')
    # Strided layer/slot storage, sentinel sibling regions and padded requests.
    initial = torch.arange(2 * 3 * 2 * 132 * 584, device=device).to(torch.uint8).reshape(2, 3, 2, 132, 584)
    storage = initial.clone()
    cache = storage[:, :, 1].view(torch.float8_e4m3fn)
    engine = object.__new__(StateCacheEngine)
    engine._named_state_caches = {'v4_window_kv_fp8': cache}
    slots = torch.tensor([1, -1, 2], device=device)
    starts = torch.tensor([130, 131, 6], device=device)
    lengths = torch.tensor([4, 2, 3], device=device)
    rejected = lengths.clone()
    offsets = torch.arange(4, device=device)[None]
    positions = (starts[:, None] + offsets) % 132
    valid = offsets < lengths[:, None]
    kv = torch.arange(2 * 12 * 512, device=device).float().sin().reshape(2, 12, 512).bfloat16()
    packed = torch.arange(2 * 12 * 584).to(torch.uint8).reshape(2, 12, 584)

    def pack(dst, keep):
        if device == 'cuda':
            from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
            for layer in range(2):
                pack_window_tokens_fp8(kv[layer], dst[layer], slots[:, None].expand(-1, 4).reshape(-1),
                                       torch.where(keep, positions, -1).reshape(-1))
        else:
            # Independent byte-layout writer; do not use the transaction's views/rows as the oracle.
            flat = dst.view(torch.uint8).view(2, 3, -1)
            for batch, slot in enumerate(slots.tolist()):
                for token in range(4):
                    if slot >= 0 and keep[batch, token]:
                        pos = int(positions[batch, token])
                        flat[:, slot, pos * 576:(pos + 1) * 576] = packed[:, batch * 4 + token, :576]
                        scale = 132 * 576 + pos * 8
                        flat[:, slot, scale:scale + 7] = packed[:, batch * 4 + token, 576:583]

    def step():
        transaction = engine.begin_v4_speculative_transaction(slots, starts, lengths, 4)
        pack(cache, valid)
        engine.finish_v4_speculative_transaction(transaction, rejected)

    for capture in ([False, True] if device == 'cuda' else [False]):
        if capture:
            step()  # Compile before capture; capture the entire snapshot/pack/rollback path.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
        for accepted in (0, 1, 4):  # Draft workspace, partial target rejection, all-accepted control.
            rejected.copy_((lengths - accepted).clamp_min(0))
            storage.copy_(initial)
            expected = initial.clone()
            pack(expected[:, :, 1].view(torch.float8_e4m3fn), valid & (offsets < accepted))
            graph.replay() if capture else step()
            torch.testing.assert_close(storage, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize('inactive', [False, True])
def test_v4_masked_restore_byte_exact_and_graph_safe(dtype, inactive, device='cuda'):
    from lmdeploy.pytorch.kernels.cuda.copy_cache import restore_state_rows

    # Use byte payloads so FP8 comparison never converts or canonicalizes NaNs.
    cache = torch.arange(2 * 3 * 8 * 12, dtype=torch.int32).to(torch.uint8).reshape(2, 3, 8, 12)
    cache = cache.view(dtype).to(device)
    slots = torch.tensor([-1, -1, -1] if inactive else [1, -1, 2], device=device)
    rows = torch.tensor([[6, 7, 0, 1], [7, 0, 1, 2], [7, 0, 1, 2]], device=device)
    rejected = torch.tensor([[False, False, True, True], [True] * 4, [False, True, True, True]], device=device)
    before = cache.view(torch.uint8)[:, slots.clamp_min(0)[:, None], rows].clone().view(dtype)
    cache.view(torch.uint8).fill_(233)
    expected = cache.view(torch.uint8).cpu().clone()
    for batch, slot in enumerate(slots.cpu().tolist()):
        for col in range(4):
            if slot >= 0 and rejected[batch, col]:
                expected[:, slot, rows[batch, col]] = before.view(torch.uint8)[:, batch, col].cpu()
    restore_state_rows(cache, before, slots, rows, rejected)
    torch.testing.assert_close(cache.view(torch.uint8).cpu(), expected)
    if device == 'cuda':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            restore_state_rows(cache, before, slots, rows, rejected)
        cache.view(torch.uint8).fill_(233)
        graph.replay()
        torch.testing.assert_close(cache.view(torch.uint8).cpu(), expected)

@pytest.mark.parametrize('layout_cls', [PackedStateCacheLayout, ContiguousStateCacheLayout])
@pytest.mark.parametrize('flags', [(True, True, True), (False, True, False), (False, False, False)])
def test_state_cache_engine_checkpoint_policy_preserves_runtime_state(monkeypatch, layout_cls, flags):
    specs = [StateCacheSpec(f'state_{i}', (2, 3), torch.float32,
                            layer_ids=[1, 3] if i == 1 else None, prefix_checkpoint=flag)
             for i, flag in enumerate(flags)]
    shapes = [(spec.shape, spec.dtype) for spec in specs]
    config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0,
                         num_gpu_blocks=0, num_state_caches=4, states_shapes=shapes)

    def allocate(tensor_specs, num_caches, device):
        return layout_cls(tuple(tensor_specs)).allocate(num_caches, 'cpu' if device == 'cuda' else device)

    monkeypatch.setattr(state_module, '_allocate_state_caches', allocate)
    engine = StateCacheEngine(config, SimpleNamespace(state_cache_specs=specs))
    assert engine.allocation.nbytes == 4 * engine.get_state_slot_nbytes(shapes, specs)
    # The middle declaration is layered; its model-facing slot axis is 1.
    views = [view.movedim(1, 0) if i == 1 else view for i, view in enumerate(engine.state_caches)]
    for i, view in enumerate(views):
        assert view.shape == ((4, 2, 2, 3) if i == 1 else (4, 2, 3))
        view.fill_(-11)
        view[0].fill_(i + 1)

    engine.copy_slots(0, 2, checkpoint_only=True)
    for i, (flag, view) in enumerate(zip(flags, views)):
        assert torch.all(view[2] == (i + 1 if flag else -11))
        view[0].fill_(-7)  # Producer reuse must not change the frozen copy.
    engine.copy_slots(2, 1, checkpoint_only=True)
    for i, (flag, view) in enumerate(zip(flags, views)):
        assert torch.all(view[1] == (i + 1 if flag else -11))

    engine.copy_slots(0, 3)  # General movement still includes runtime-only state.
    for view in views:
        assert torch.all(view[3] == -7)
    engine.zero_slots(torch.tensor([0, 3]), torch.tensor([True, False]))
    for view in views:
        assert torch.count_nonzero(view[0]) == 0
        assert torch.all(view[3] == -7)
