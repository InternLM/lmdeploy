# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.cuda.mbgmm import mbgmm_a, mbgmm_b
from lmdeploy.pytorch.kernels.cuda.mbgmv import mbgmv_a, mbgmv_b
from lmdeploy.pytorch.kernels.rearange_all_gather import rearange_all_gather
from lmdeploy.pytorch.model_inputs import StepContextManager

from ..slora import AdapterInfo, SLoRABuilder, SLoRAImpl


@dataclass
class PackedLoRAInput:
    """packed lora input."""
    x: torch.Tensor
    q_start_loc: torch.Tensor
    q_seqlens: torch.Tensor
    adapter_ids: torch.Tensor
    max_seq_len: int
    is_decoding: bool


class TritonSLoRAImpl(SLoRAImpl):
    """triton slora implementation."""

    @staticmethod
    def _make_packed_lora_input(x, ctx_mgr):
        """make PackedLoRAInput."""
        context = ctx_mgr.current_context()

        # adapter cache
        max_q_seq_length = x.numel() // x.size(-1)

        return PackedLoRAInput(x=x.flatten(0, -2).contiguous(),
                               q_start_loc=context.q_start_loc,
                               q_seqlens=context.q_seqlens,
                               adapter_ids=context.local_adapter_ids,
                               max_seq_len=max_q_seq_length,
                               is_decoding=context.is_decoding)

    def _forward_rowwise(self,
                         lora_input: PackedLoRAInput,
                         base_output: torch.Tensor,
                         adapter_info: AdapterInfo,
                         is_tp: bool = True):
        """forward_rowwise."""
        sliced_base = base_output[..., adapter_info.base_slice]
        out_size = sliced_base.size(-1)
        if is_tp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            out_size //= world_size
        if not lora_input.is_decoding:
            xa = mbgmm_a(lora_input.x,
                         adapter_info.a_cache,
                         q_start_loc=lora_input.q_start_loc,
                         q_seqlens=lora_input.q_seqlens,
                         adapter_ids=lora_input.adapter_ids,
                         rank_offset=adapter_info.rank_offsets,
                         ranks=adapter_info.ranks,
                         max_seq_len=lora_input.max_seq_len,
                         max_rank=adapter_info.max_rank)
            lora_out = mbgmm_b(xa,
                               adapter_info.b_cache,
                               q_start_loc=lora_input.q_start_loc,
                               q_seqlens=lora_input.q_seqlens,
                               adapter_ids=lora_input.adapter_ids,
                               scaling=adapter_info.scalings,
                               rank_offset=adapter_info.rank_offsets,
                               ranks=adapter_info.ranks,
                               max_seq_len=lora_input.max_seq_len,
                               max_rank=adapter_info.max_rank,
                               out_size=out_size)
        else:
            xa = mbgmv_a(lora_input.x,
                         adapter_info.a_cache,
                         adapter_ids=lora_input.adapter_ids,
                         rank_offset=adapter_info.rank_offsets,
                         ranks=adapter_info.ranks,
                         max_rank=adapter_info.max_rank)
            lora_out = mbgmv_b(xa,
                               adapter_info.b_cache,
                               adapter_ids=lora_input.adapter_ids,
                               scaling=adapter_info.scalings,
                               rank_offset=adapter_info.rank_offsets,
                               ranks=adapter_info.ranks,
                               max_rank=adapter_info.max_rank,
                               out_size=out_size)

        if is_tp:
            out_shape = base_output.shape
            out = base_output.flatten(0, -2)
            slice_off = adapter_info.base_slice.start
            slice_off = 0 if slice_off is None else slice_off
            slice_start = slice_off + rank * out_size
            slice_end = slice_start + out_size
            out[:, slice_start:slice_end] += lora_out
            out = out.reshape(out_shape)
        else:
            lora_out = lora_out.reshape(sliced_base.shape)
            sliced_base.add_(lora_out)
            out = base_output

        return out

    def _forward_colwise(
        self,
        lora_input: PackedLoRAInput,
        base_output: torch.Tensor,
        adapter_info: AdapterInfo,
    ):
        """forward_colwise."""

        def __gather_xa(xa):
            """gather xa."""
            gathered_xa = xa.new_empty(world_size, xa.size(0), xa.size(1))
            dist.all_gather_into_tensor(gathered_xa, xa)
            # TODO: gather would failed when adapters have different ranks.
            gathered_xa = gathered_xa.permute(1, 0, 2).flatten(-2, -1)
            return gathered_xa

        base_slice = adapter_info.base_slice
        a_cache = adapter_info.a_cache
        b_cache = adapter_info.b_cache
        rank_offsets = adapter_info.rank_offsets
        ranks = adapter_info.ranks
        max_rank = adapter_info.max_rank
        scalings = adapter_info.scalings
        sliced_base = base_output[..., base_slice]
        out_size = sliced_base.size(-1)
        world_size = dist.get_world_size()

        if not lora_input.is_decoding:
            xa = mbgmm_a(lora_input.x,
                         a_cache,
                         q_start_loc=lora_input.q_start_loc,
                         q_seqlens=lora_input.q_seqlens,
                         adapter_ids=lora_input.adapter_ids,
                         rank_offset=rank_offsets,
                         ranks=ranks,
                         max_seq_len=lora_input.max_seq_len,
                         max_rank=max_rank,
                         rank_step=world_size)
            gathered_xa = __gather_xa(xa)
            if len(ranks) > 1:
                gathered_xa = rearange_all_gather(
                    gathered_xa,
                    b_start_loc=lora_input.q_start_loc,
                    b_seq_lens=lora_input.q_seqlens,
                    adapter_ids=lora_input.adapter_ids,
                    ranks=ranks,
                    world_size=world_size,
                    max_seq_len=lora_input.max_seq_len,
                    output=gathered_xa)
            lora_out = mbgmm_b(gathered_xa,
                               b_cache,
                               q_start_loc=lora_input.q_start_loc,
                               q_seqlens=lora_input.q_seqlens,
                               adapter_ids=lora_input.adapter_ids,
                               scaling=scalings,
                               rank_offset=rank_offsets,
                               ranks=ranks,
                               max_seq_len=lora_input.max_seq_len,
                               max_rank=max_rank,
                               out_size=out_size)
        else:
            xa = mbgmv_a(lora_input.x,
                         a_cache,
                         adapter_ids=lora_input.adapter_ids,
                         rank_offset=rank_offsets,
                         ranks=ranks,
                         max_rank=max_rank,
                         rank_step=world_size)
            gathered_xa = __gather_xa(xa)
            if len(ranks) > 1:
                gathered_xa = rearange_all_gather(
                    gathered_xa,
                    b_start_loc=lora_input.q_start_loc,
                    b_seq_lens=lora_input.q_seqlens,
                    adapter_ids=lora_input.adapter_ids,
                    ranks=ranks,
                    world_size=world_size,
                    max_seq_len=lora_input.max_seq_len,
                    output=gathered_xa)
            lora_out = mbgmv_b(gathered_xa,
                               b_cache,
                               adapter_ids=lora_input.adapter_ids,
                               scaling=scalings,
                               rank_offset=rank_offsets,
                               ranks=ranks,
                               max_rank=max_rank,
                               out_size=out_size)

        lora_out = lora_out.reshape(sliced_base.shape)
        sliced_base.add_(lora_out)
        output = base_output
        return output

    def forward(self,
                x: torch.Tensor,
                base_output: torch.Tensor,
                adapter_info: AdapterInfo,
                ctx_mgr: StepContextManager,
                colwise: bool,
                is_tp: bool = True):
        """forward."""
        lora_input = self._make_packed_lora_input(x, ctx_mgr)
        if colwise and is_tp:
            return self._forward_colwise(lora_input, base_output, adapter_info)
        else:
            return self._forward_rowwise(lora_input, base_output, adapter_info,
                                         is_tp)


class TritonSLoRABuilder(SLoRABuilder):
    """triton slora layer builder."""

    @staticmethod
    def build():
        """build."""
        return TritonSLoRAImpl()
