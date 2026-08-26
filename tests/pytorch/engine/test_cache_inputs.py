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
