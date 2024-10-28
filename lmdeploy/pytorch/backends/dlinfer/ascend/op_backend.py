# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class AscendOpsBackend(DlinferOpsBackend):
    """ascend layer backend."""
    enable_graph = False
    half_negative_inf = torch.finfo(torch.float16).min
    total_slots = None

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

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(
                    block_num * block_size,
                    dtype=torch.long,
                    device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices, attention_mask = [], []
        block_num, block_size, _ = step_context.kv_caches[0][0].shape
        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())
        q_seqlens_list = step_context.q_seqlens.tolist()
        kv_seqlens_list = step_context.kv_seqlens.tolist()
        max_q_seq_len = max(q_seqlens_list)
        max_kv_seq_len = max(kv_seqlens_list)

        for i in range(step_context.q_start_loc.size(0)):
            q_seq_len = q_seqlens_list[i]
            kv_seq_len = kv_seqlens_list[i]

            # collect kv start indices.
            history_length = kv_seq_len - q_seq_len
            total_slots = get_total_slots()
            slot_tables = total_slots[step_context.block_offsets[i]].view(-1)
            slots = slot_tables[history_length:kv_seq_len]
            kv_start_indices.append(slots)

            # collect attention mask of paged_prefill attention stage.
            if not (step_context.is_decoding or is_unpaged_prefill):
                single_attention_mask = torch.logical_not(
                    torch.tril(
                        torch.ones(q_seq_len,
                                   step_context.block_offsets.shape[1] *
                                   block_size,
                                   dtype=torch.bool,
                                   device=step_context.block_offsets.device),
                        diagonal=kv_seq_len - q_seq_len,
                    ))
                attention_mask.append(single_attention_mask)

        kv_start_indices = torch.cat(kv_start_indices)

        if step_context.is_decoding:
            # prepare some params of paged_decode attention stage.
            q_start_loc_cpu, q_seqlens_cpu = None, None
        elif is_unpaged_prefill:
            # prepare some params of unpaged_prefill attention stage.
            q_start_loc_cpu, kv_seqlens_cpu = None, None
            q_seqlens_cpu = step_context.q_seqlens.cpu()
            single_attention_mask = torch.logical_not(
                torch.tril(
                    torch.ones(max_q_seq_len, max_kv_seq_len,
                               dtype=torch.bool).cuda(),
                    diagonal=max_kv_seq_len - max_q_seq_len,
                ))
            attention_mask.append(single_attention_mask)
        else:
            # prepare some params of paged_prefill attention stage.
            q_start_loc_cpu, q_seqlens_cpu = None, None
            attention_mask = [torch.cat([mask for mask in attention_mask])]

        if cls.enable_graph:
            kv_start_indices = kv_start_indices.view(-1).to(torch.int32)
            import torch._dynamo as dynamo
            if not is_unpaged_prefill:
                step_context.block_offsets = step_context.block_offsets.to(
                    torch.int32)
                if not step_context.is_decoding:
                    step_context.block_offsets = step_context.block_offsets\
                        .repeat_interleave(step_context.q_seqlens, 0)
            dynamo.mark_dynamic(step_context.block_offsets, [0, 1])
            kv_seqlens = step_context.kv_seqlens.to(torch.int32)
            if not step_context.is_decoding:
                if is_unpaged_prefill:
                    attention_mask = [mask.half() for mask in attention_mask]
                else:
                    attention_mask = [
                        torch.cat([
                            mask.half() * cls.half_negative_inf
                            for mask in attention_mask
                        ]).unsqueeze(1)
                    ]
                    kv_seqlens = kv_seqlens.repeat_interleave(
                        step_context.q_seqlens, 0)
        else:
            if step_context.is_decoding:
                kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            elif is_unpaged_prefill:
                pass
            else:
                kv_seqlens_cpu = step_context.kv_seqlens.repeat_interleave(
                    step_context.q_seqlens, 0).cpu()
                block_offsets_int32 = step_context.block_offsets.to(
                    torch.int32)
                step_context.block_offsets = block_offsets_int32\
                    .repeat_interleave(step_context.q_seqlens, 0)
            kv_seqlens = kv_seqlens_cpu

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc_cpu,
            q_seqlens=q_seqlens_cpu,
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
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from .graph_runner import AscendGraphRunner
        ascend_graph_runner = AscendGraphRunner(model, model_config,
                                                cache_config, backend_config,
                                                device)
        AscendOpsBackend.enable_graph = ascend_graph_runner.enable_graph
        return ascend_graph_runner
