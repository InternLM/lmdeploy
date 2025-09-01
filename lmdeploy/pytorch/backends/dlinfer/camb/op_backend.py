# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class CambOpsBackend(DlinferOpsBackend):
    """Camb layer backend."""
    total_slots = None

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'camb'

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            num_heads,
            block_size,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            num_heads,
            block_size,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(block_num * block_size,
                                               dtype=torch.int32,
                                               device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices = []
        block_num, _, block_size, _ = step_context.kv_caches[0][0].shape

        is_unpaged_prefill = False
        q_start_loc = step_context.q_start_loc
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens.to(torch.int32)
        block_offsets = step_context.block_offsets.to(torch.int32)
        max_q_seq_len = torch.max(q_seqlens).cpu().item()
        max_kv_seq_len = torch.max(kv_seqlens).cpu().item()

        cu_seqlens = torch.cat((q_start_loc, q_seqlens.sum().unsqueeze(0))).int()
        cu_seq_lens_kv = None

        q_seqlens_list = step_context.q_seqlens.tolist()
        kv_seqlens_list = step_context.kv_seqlens.tolist()
        if not step_context.is_decoding:
            is_unpaged_prefill = q_seqlens_list == kv_seqlens_list
            # get kv_indices
            for i in range(q_start_loc.size(0)):
                q_seq_len = q_seqlens_list[i]
                kv_seq_len = kv_seqlens_list[i]
                # collect kv start indices.
                history_length = kv_seq_len - q_seq_len
                total_slots = get_total_slots()
                slot_tables = total_slots[block_offsets[i]].view(-1)
                slots = slot_tables[history_length:kv_seq_len]
                kv_start_indices.append(slots)
            kv_start_indices = torch.cat(kv_start_indices)
            if not is_unpaged_prefill:
                cu_seq_lens_kv = torch.cat((torch.tensor([0], device=kv_seqlens.device), kv_seqlens.cumsum(0))).int()
        else:
            # collect kv_start_indices without using a for-loop,
            # (fill kv-cache for just ONE token during the decoding phase)
            idx = (step_context.kv_seqlens - 1) % block_size
            block_num = (step_context.kv_seqlens - 1) // block_size
            last_block = block_offsets.gather(  # dtype of gather must be int64
                1, block_num.view(-1, 1)).view(-1)
            kv_start_indices = (last_block * block_size + idx).to(torch.int32)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            block_offsets,
            q_start_loc=cu_seqlens,
            cu_seq_lens_kv=cu_seq_lens_kv,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=None,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
        )

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner
        return CUDAGraphRunner(model, model_config, cache_config, backend_config, device)

    @staticmethod
    def support_ray():
        """Support ray."""
        return True
