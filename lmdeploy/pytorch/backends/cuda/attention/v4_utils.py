# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for DeepSeek-V4 model and backend implementations."""

from dataclasses import dataclass

import torch


@dataclass
class V4PrefillTokenMeta:
    """Per-token sequence mapping for one prefill step."""

    token_pos: torch.Tensor
    seq_id: torch.Tensor


def build_prefill_token_meta(q_seqlens: torch.Tensor,
                             cu_q_seqlens: torch.Tensor | None = None):
    """Build per-token prefill sequence mapping without CUDA sync.

    Given q_seqlens [bsz], returns token_pos [total_q] where token_pos[i] is the position of token i within its
    sequence, plus seq_id [total_q].
    """
    if cu_q_seqlens is None:
        cu_q_seqlens = torch.cat([
            q_seqlens.new_zeros(1, device=q_seqlens.device),
            q_seqlens.cumsum(0),
        ])
    total_tokens = cu_q_seqlens[-1]
    token_id = torch.arange(total_tokens, dtype=cu_q_seqlens.dtype,
                            device=q_seqlens.device)
    seq_id = torch.searchsorted(cu_q_seqlens[1:], token_id, right=True)
    token_pos = token_id - cu_q_seqlens[seq_id]
    return V4PrefillTokenMeta(
        token_pos=token_pos,
        seq_id=seq_id,
    )
