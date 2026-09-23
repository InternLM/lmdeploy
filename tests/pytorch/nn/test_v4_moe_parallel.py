# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.config import DistConfig
from lmdeploy.pytorch.nn.moe import base, v4_fp4


@pytest.mark.parametrize('dp,tp,ep', [(1, 2, 1), (2, 4, 1), (2, 1, 2)])
def test_v4_fp4_gathers_dp_rows_before_tp_experts(monkeypatch, dp, tp, ep):
    """TP experts must see the same gathered rows that reduce-scatter
    splits."""
    config = DistConfig(dp=dp, tp=tp, ep=ep)
    monkeypatch.setattr(base, 'get_dist_manager', lambda: SimpleNamespace(current_config=lambda: config))
    monkeypatch.setattr(base.dist, 'get_dist_manager', lambda: SimpleNamespace(current_config=lambda: config))
    monkeypatch.setattr(base, 'get_step_ctx_manager', lambda: SimpleNamespace(
        current_context=lambda: SimpleNamespace(dp_meta=SimpleNamespace(moe_tp_sizes=[2, 3]))))
    gather_group, tp_group = object(), object()
    local = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.long)
    values = [local, weights, ids]
    global_values = [torch.cat([v, v.new_full((3, v.size(1)), 9)]) for v in values]
    calls = []

    def gather(value, sizes, group=None, **kwargs):
        assert group is gather_group and sizes == [2, 3]
        index = len(calls)
        calls.append('gather')
        torch.testing.assert_close(value, values[index])
        return global_values[index]

    def reduce_scatter(output, shards, group=None, **kwargs):
        assert group is tp_group
        assert [x.size(0) for x in shards] == [2, 2, 3, 3]
        calls.append('reduce_scatter')
        output.mul_(config.moe_tp)

    def all_reduce(output, group=None, **kwargs):
        assert group is tp_group
        calls.append('all_reduce')
        output.mul_(config.moe_tp)

    monkeypatch.setattr(base.dist, 'gather_by_tp_sizes', gather)
    monkeypatch.setattr(base.dist.dist, 'reduce_scatter', reduce_scatter)
    monkeypatch.setattr(base.dist, 'all_reduce', all_reduce)

    def gemm(hidden, route_weights, route_ids, *args):
        expected = global_values if dp > 1 and ep == 1 else values
        for actual, ref in zip((hidden, route_weights, route_ids), expected):
            torch.testing.assert_close(actual, ref)
        calls.append('gemm')
        return hidden.clone() / (config.moe_tp if ep == 1 else 1)

    model = v4_fp4.FusedMoEV4FP4.__new__(v4_fp4.FusedMoEV4FP4)
    nn.Module.__init__(model)
    model.ep_size = ep
    model.tp_rank = 0
    model.tp_mode = config.moe_tp_mode
    model.tp_group = tp_group
    model.gather_group = gather_group
    model.impl = SimpleNamespace(forward=gemm)
    model.gate_up = model.down = SimpleNamespace(weight=None, scale=None)
    output = model(local, weights, ids)
    torch.testing.assert_close(output, local)
    expected_calls = ['gemm'] if ep > 1 else (
        ['gather'] * 3 + ['gemm', 'reduce_scatter'] if dp > 1 else ['gemm', 'all_reduce'])
    assert calls == expected_calls
