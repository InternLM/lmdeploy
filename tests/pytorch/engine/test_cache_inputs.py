# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs


def test_cache_checkpoint_inputs_to_device_moves_only_kv_plans():
    kv_restore_plan = torch.tensor([[1, 2], [3, 4]])
    kv_save_plan = torch.tensor([[5], [6]])
    state_restore_plan = ((7, 8), (9, 10))
    state_save_plan = ((11, ), (12, ))
    cache_inputs = CacheCheckpointInputs(
        kv_restore_plan=kv_restore_plan,
        kv_save_plan=kv_save_plan,
        state_restore_plan=state_restore_plan,
        state_save_plan=state_save_plan,
    )

    device_inputs = cache_inputs.to_device('meta', non_blocking=True)

    assert device_inputs is not cache_inputs
    assert device_inputs.kv_restore_plan.device.type == 'meta'
    assert device_inputs.kv_save_plan.device.type == 'meta'
    assert device_inputs.state_restore_plan is state_restore_plan
    assert device_inputs.state_save_plan is state_save_plan


def test_cache_checkpoint_inputs_record_stream_records_only_kv_plans():
    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append((id(self), stream))

    stream = object()
    restore_plan = _CudaTensor()
    save_plan = _CudaTensor()
    cache_inputs = CacheCheckpointInputs(
        kv_restore_plan=restore_plan,
        kv_save_plan=save_plan,
        state_restore_plan=((1, ), (2, )),
        state_save_plan=((3, ), (4, )),
    )

    cache_inputs.record_stream(stream)

    assert recorded == [
        (id(restore_plan), stream),
        (id(save_plan), stream),
    ]


def test_compact_one_forward_layout_transfer_pickle_and_copy_exclusion():
    import pickle

    import pytest

    from lmdeploy.pytorch.compact_state import CompactStatePrefill
    plan = CompactStatePrefill.from_lengths([72], [2048], checkpoint_ids=[7], checkpoint_offsets=[64])
    inputs = CacheCheckpointInputs(compact_prefill=plan, kv_save_plan=torch.tensor([[1], [2]]))
    restored = pickle.loads(pickle.dumps(inputs))
    assert restored.compact_prefill.checkpoint_ids.tolist() == [7]
    assert restored.compact_prefill.checkpoint_offsets.tolist() == [64]
    moved = restored.to_device('meta')
    assert moved.compact_prefill.cu_seqlens.device.type == 'meta'
    assert moved.compact_prefill.checkpoint_ids.device.type == 'meta'
    assert moved.kv_save_plan.device.type == 'meta'
    for field in ('state_restore_plan', 'state_save_plan'):
        with pytest.raises(ValueError, match='legacy state-copy'):
            CacheCheckpointInputs(compact_prefill=plan, **{field: ((1,), (2,))})


def test_compact_source_binding_serializes_moves_and_excludes_legacy_copy():
    import pickle

    import pytest

    sources = torch.tensor([-1, 0, 7])
    packet = pickle.loads(pickle.dumps(CacheCheckpointInputs(compact_source_ids=sources)))
    assert packet.compact_source_ids.tolist() == [-1, 0, 7]
    assert packet.to_device('meta').compact_source_ids.device.type == 'meta'
    with pytest.raises(ValueError, match='legacy state-copy'):
        CacheCheckpointInputs(compact_source_ids=sources, state_restore_plan=((1,), (2,)))
    for invalid in (sources.int(), sources[None], sources[::2]):
        with pytest.raises(ValueError, match='contiguous per-request int64'):
            CacheCheckpointInputs(compact_source_ids=invalid)
