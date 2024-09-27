# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class AscendOpsBackend(DefaultOpsBackend):
    """ascend layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'ascend'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get ascend layer builder."""
        if layer_type == OpType.Attention:
            from .attention import AscendAttentionBuilder
            return AscendAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import AscendApplyRotaryEmbBuilder
            return AscendApplyRotaryEmbBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import AscendRMSNormBuilder
            return AscendRMSNormBuilder
        elif layer_type == OpType.SoftmaxTopK:
            from .moe import AscendSoftmaxTopKBuilder
            return AscendSoftmaxTopKBuilder
        elif layer_type == OpType.FusedMoE:
            from .moe import AscendFusedMoEBuilder
            return AscendFusedMoEBuilder
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        from .attention import AscendAttentionMetadata
        return AscendAttentionMetadata

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

        is_unpaged_prefill = False
        q_start_loc_cpu = step_context.q_start_loc.cpu()
        q_seqlens_cpu = step_context.q_seqlens.cpu()
        kv_seqlens_cpu = step_context.kv_seqlens.cpu()
        max_q_seq_len = torch.max(q_seqlens_cpu).item()
        max_kv_seq_len = torch.max(kv_seqlens_cpu).item()

        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())
            if is_unpaged_prefill:
                single_attention_mask = torch.logical_not(
                    torch.tril(
                        torch.ones(max_q_seq_len,
                                   max_kv_seq_len,
                                   dtype=torch.bool).cuda(),
                        diagonal=max_kv_seq_len - max_q_seq_len,
                    ))
                attention_mask.append(single_attention_mask)
        total_slots = torch.arange(block_num * block_size,
                                   dtype=torch.long,
                                   device=device)
        total_slots = total_slots.view(block_num, block_size)
        for i in range(step_context.q_start_loc.size(0)):
            q_seq_len = int(step_context.q_seqlens[i])
            kv_seq_len = int(step_context.kv_seqlens[i])
            if not (step_context.is_decoding or is_unpaged_prefill):
                single_attention_mask = torch.logical_not(
                    torch.tril(
                        torch.ones(step_context.q_seqlens[i],
                                   step_context.block_offsets.shape[1] *
                                   block_size,
                                   dtype=torch.bool).cuda(),
                        diagonal=step_context.kv_seqlens[i] -
                        step_context.q_seqlens[i],
                    ))
                attention_mask.append(single_attention_mask)
            history_length = kv_seq_len - q_seq_len
            slot_tables = total_slots[step_context.block_offsets[i]].flatten()
            slot_indices = [p for p in range(history_length, kv_seq_len)]
            slots = slot_tables[slot_indices].reshape((-1, 1))
            kv_start_indices.append(slots)
        kv_start_indices = torch.cat(kv_start_indices)

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
