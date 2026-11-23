# Copyright (c) 2024, DeepLink. All rights reserved.
import torch


def ascend_chatglm2_fill_rope(states: torch.Tensor, rope: torch.Tensor):
    """Fill rope."""
    rope_part = states.chunk(2, -1)[1]
    rope = rope.unflatten(-1, (2, -1))
    rope = rope.transpose(-2, -1).flatten(-2, -1)
    states = torch.cat([rope_part, rope], dim=-1)

    return states
