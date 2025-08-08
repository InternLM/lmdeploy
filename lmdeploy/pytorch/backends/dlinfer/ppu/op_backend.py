# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class PpuOpsBackend(DlinferOpsBackend):
    """PPU layer backend."""
    total_slots = None

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'ppu'

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(block_num * block_size,
                                               dtype=torch.long,
                                               device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices, attention_mask = [], []
        block_num, block_size, _, _ = step_context.kv_caches[0][1].shape

        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = torch.all(step_context.q_seqlens.eq(step_context.kv_seqlens))

        q_start_loc = torch.cat(
            (step_context.q_start_loc, (step_context.q_start_loc[-1] + step_context.q_seqlens[-1]).unsqueeze(0)))
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens
        max_q_seq_len = torch.max(q_seqlens)
        max_kv_seq_len = torch.max(kv_seqlens)

        if step_context.is_decoding:
            idx = (step_context.kv_seqlens - 1) % block_size
            b_num = (step_context.kv_seqlens - 1) // block_size
            last_block = step_context.block_offsets.gather(1, b_num.view(-1, 1)).view(-1)
            kv_start_indices = (last_block * block_size + idx).reshape((-1, 1))
        else:
            for i in range(step_context.q_start_loc.size(0)):
                q_seq_len = int(step_context.q_seqlens[i])
                kv_seq_len = int(step_context.kv_seqlens[i])
                history_length = kv_seq_len - q_seq_len
                total_slots = get_total_slots()
                slot_tables = total_slots[step_context.block_offsets[i]].view(-1)
                slots = slot_tables[history_length:kv_seq_len]
                kv_start_indices.append(slots)
            kv_start_indices = torch.cat(kv_start_indices)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
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

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner
        return CUDAGraphRunner(model, model_config, cache_config, backend_config, device)
