# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayState


@pytest.mark.parametrize('family', ['qwen3_next', 'qwen3_5'])
@pytest.mark.parametrize('indexed_replay', [False, True])
def test_qwen_model_passes_complete_indexed_replay_to_conv(family, indexed_replay):
    from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5GatedDeltaNet
    from lmdeploy.pytorch.models.qwen3_next import Qwen3NextGatedDeltaNet

    value = torch.zeros(1, 1, 128)
    state = GatedDeltaReplayState(value, value, value, value, value, None,
                                  shared_conv_state=value if indexed_replay else None)
    expected = state if indexed_replay else value

    class ReachedConv(Exception):
        pass

    def conv(x, conv_state, metadata):
        assert conv_state is expected
        raise ReachedConv

    model = SimpleNamespace(in_proj_qkv=lambda x: value, in_proj_z=lambda x: value,
                             in_proj_ba=lambda x: value, in_proj_qkvz=lambda x: value,
                             head_v_dim=128, fix_ba_ordering=lambda x: (value, value),
                             fix_query_key_value_ordering=lambda *args: (value, value, value, value), conv1d=conv)
    cls = Qwen3NextGatedDeltaNet if family == 'qwen3_next' else Qwen3_5GatedDeltaNet
    with pytest.raises(ReachedConv):
        cls.forward(model, value, state, None)
