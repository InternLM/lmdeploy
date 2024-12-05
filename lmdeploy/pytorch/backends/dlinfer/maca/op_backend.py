# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class MacaOpsBackend(DlinferOpsBackend):
    """maca layer backend."""
    total_slots = None

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'maca'

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        x = 16
        return (num_heads, head_size // x, block_size, x)

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (num_heads, block_size, head_size)

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(
                    block_num * block_size,
                    dtype=torch.long,
                    device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices, attention_mask = [], []
        block_num, _, block_size, _ = step_context.kv_caches[0][1].shape
        device = step_context.block_offsets.device

        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = \
               all((step_context.q_seqlens ==
                    step_context.kv_seqlens).tolist())
        q_start_loc = torch.cat((torch.tensor([0], device=device),
                                 step_context.q_seqlens.cumsum(0))).int()
        q_seqlens = step_context.q_seqlens.int()
        kv_seqlens = step_context.kv_seqlens.int()
        max_q_seq_len = torch.max(q_seqlens).item()
        max_kv_seq_len = torch.max(kv_seqlens).item()

        if step_context.is_decoding:
            # collect kv_start_indices without using a for-loop,
            # (fill kv-cache for just ONE token during the decoding phase)
            idx = (step_context.kv_seqlens - 1) % block_size
            b_num = (step_context.kv_seqlens - 1) // block_size
            last_block = step_context.block_offsets.gather(
                1, b_num.view(-1, 1)).view(-1)
            kv_start_indices = (last_block * block_size + idx).reshape((-1, 1))
        else:
            for i in range(step_context.q_start_loc.size(0)):
                q_seq_len = int(step_context.q_seqlens[i])
                kv_seq_len = int(step_context.kv_seqlens[i])
                # collect kv start indices during the prefill phase.
                history_length = kv_seq_len - q_seq_len
                total_slots = get_total_slots()
                slot_tables = total_slots[step_context.block_offsets[i]].view(
                    -1)
                slots = slot_tables[history_length:kv_seq_len]
                kv_start_indices.append(slots)
            kv_start_indices = torch.cat(kv_start_indices)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets.int(),
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
        )

        step_context.attn_metadata = attn_metadata
        return step_context
