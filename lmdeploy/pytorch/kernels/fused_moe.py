# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch

from .dispatcher import FunctionDispatcher


def _fused_moe_api(hidden_states: torch.Tensor,
                   w1: torch.Tensor,
                   w2: torch.Tensor,
                   topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor,
                   topk: int,
                   renormalize: bool = False) -> torch.Tensor:
    """fused moe."""
    ...


fused_moe = FunctionDispatcher('fused_moe').make_caller(_fused_moe_api)
