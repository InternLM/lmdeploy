# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

import lmdeploy.pytorch.distributed as distributed


def test_reduce_scatter_by_tp_sizes_uses_disjoint_output(monkeypatch):
    """The collective output must not alias any reduce-scatter input view."""
    manager = SimpleNamespace(current_config=lambda: SimpleNamespace(attn_tp=2))
    monkeypatch.setattr(distributed, 'get_dist_manager', lambda: manager)

    seen = {}

    def _reduce_scatter(output, inputs, group):
        seen['group'] = group
        seen['inputs'] = inputs
        output_storage = output.untyped_storage().data_ptr()
        assert all(output_storage != item.untyped_storage().data_ptr() for item in inputs)
        output.copy_(inputs[0])

    monkeypatch.setattr(distributed.dist, 'reduce_scatter', _reduce_scatter)

    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    result = distributed.reduce_scatter_by_tp_sizes(source, rank=0, tp_sizes=[2, 1], group='test-group')

    assert seen['group'] == 'test-group'
    assert len(seen['inputs']) == 4
    torch.testing.assert_close(result, source[:2])
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
