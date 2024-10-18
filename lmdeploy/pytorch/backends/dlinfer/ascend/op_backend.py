# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class AscendOpsBackend(DlinferOpsBackend):
    """ascend layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'ascend'

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads * head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads * head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        block_num, block_size, _ = step_context.kv_caches[0][0].shape
        device = step_context.block_offsets.device

        q_start_loc_cpu = step_context.q_start_loc.cpu()
        q_seqlens_cpu = step_context.q_seqlens.cpu()
        max_q_seq_len = torch.max(q_seqlens_cpu).item()

        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())

        total_slots = torch.arange(block_num * block_size,
                                   dtype=torch.long,
                                   device=device)
        total_slots = total_slots.view(block_num, block_size)

        q_seqlens_list = step_context.q_seqlens.tolist()
        kv_seqlens_list = step_context.kv_seqlens.tolist()
        for i in range(step_context.q_start_loc.size(0)):
            q_seq_len = q_seqlens_list[i]
            kv_seq_len = kv_seqlens_list[i]

            # collect kv start indices.
            history_length = kv_seq_len - q_seq_len
            slot_tables = total_slots[step_context.block_offsets[i]].flatten()
            slot_indices = [p for p in range(history_length, kv_seq_len)]
            slots = slot_tables[slot_indices].reshape((-1, 1))
            kv_start_indices.append(slots)

            # collect attention mask of paged_prefill attention stage.
            if not (step_context.is_decoding or is_unpaged_prefill):
                single_attention_mask = torch.logical_not(
                    torch.tril(
                        torch.ones(q_seq_len,
                                   step_context.block_offsets.shape[1] *
                                   block_size,
                                   dtype=torch.bool).cuda(),
                        diagonal=kv_seq_len - q_seq_len,
                    ))
                attention_mask.append(single_attention_mask)

        kv_start_indices = torch.cat(kv_start_indices)

        if step_context.is_decoding:
            # calculate somae params of paged_decode attention stage.
            kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            max_kv_seq_len = torch.max(kv_seqlens_cpu).item()
        elif is_unpaged_prefill:
            # calculate somae params of unpaged_prefill attention stage.
            kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            max_kv_seq_len = torch.max(kv_seqlens_cpu).item()
            single_attention_mask = torch.logical_not(
                torch.tril(
                    torch.ones(max_q_seq_len, max_kv_seq_len,
                               dtype=torch.bool).cuda(),
                    diagonal=max_kv_seq_len - max_q_seq_len,
                ))
            attention_mask.append(single_attention_mask)
        else:
            # calculate somae params of paged_prefill attention stage.
            kv_seqlens_cpu = step_context.kv_seqlens.repeat_interleave(
                step_context.q_seqlens, 0).cpu()
            max_kv_seq_len = torch.max(kv_seqlens_cpu).item()
            block_offsets_int32 = step_context.block_offsets.to(torch.int32)
            step_context.block_offsets = block_offsets_int32.repeat_interleave(
                step_context.q_seqlens, 0)
            attention_mask = [
                torch.cat([mask for mask in attention_mask]).unsqueeze(1)
            ]

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc_cpu,
            q_seqlens=q_seqlens_cpu,
            kv_seqlens=kv_seqlens_cpu,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
        )

        step_context.attn_metadata = attn_metadata
        return step_context
