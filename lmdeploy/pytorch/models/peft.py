# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..kernels.mbgmm import mbgmm_a, mbgmm_b
from ..kernels.mbgmv import mbgmv_a, mbgmv_b
from ..kernels.rearange_all_gather import rearange_all_gather


@dataclass
class PackedLoRAInput:
    x: torch.Tensor
    a_cache: torch.Tensor
    b_cache: torch.Tensor
    b_start_loc: torch.Tensor
    b_seq_lens: torch.Tensor
    b_adapter_ids: torch.Tensor
    b_scaling: torch.Tensor
    rank_page_table: torch.Tensor
    rank_page_start: torch.Tensor
    ranks: torch.Tensor
    max_seq_len: int
    max_rank: int
    is_decoding: bool


class LoRALinear(torch.nn.Module):

    def _make_packed_lora_input(self, x):
        context = self.context.context

        # adapter cache
        global_adapter_ids = context.global_adapter_ids
        layer_idx = self.layer_idx
        ranks = self.ranks[global_adapter_ids]
        block_starts = self.block_starts[global_adapter_ids]
        k_cache, v_cache = context.kv_caches[layer_idx]
        cache_len = k_cache.size(0)
        a_cache = k_cache.view(cache_len, -1)
        b_cache = v_cache.view(cache_len, -1)

        return PackedLoRAInput(x=x.flatten(0, -2).contiguous(),
                               a_cache=a_cache,
                               b_cache=b_cache,
                               b_start_loc=context.q_start_loc,
                               b_seq_lens=context.seq_length,
                               b_adapter_ids=context.local_adapter_ids,
                               b_scaling=context.local_adapter_scalings,
                               rank_page_table=context.adapter_offsets,
                               rank_page_start=block_starts,
                               ranks=ranks,
                               max_seq_len=context.max_seq_length,
                               max_rank=context.max_rank,
                               is_decoding=context.is_decoding)

    def _lora_forward_local(self, x):
        """lora forward no tp."""

        lora_input = self._make_packed_lora_input(x)

        out_size = self.base_layer.weight.size(0)
        if not lora_input.is_decoding:
            xa = mbgmm_a(lora_input.x,
                         lora_input.a_cache,
                         b_start_loc=lora_input.b_start_loc,
                         b_seq_lens=lora_input.b_seq_lens,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_seq_len=lora_input.max_seq_len,
                         max_rank=lora_input.max_rank)
            lora_out = mbgmm_b(xa,
                               lora_input.b_cache,
                               b_start_loc=lora_input.b_start_loc,
                               b_seq_lens=lora_input.b_seq_lens,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_seq_len=lora_input.max_seq_len,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)
        else:
            xa = mbgmv_a(lora_input.x,
                         lora_input.a_cache,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_rank=lora_input.max_rank)
            lora_out = mbgmv_b(xa,
                               lora_input.b_cache,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)

        base_out = self.base_layer(x)
        lora_out = lora_out.reshape(base_out.shape)
        output = base_out + lora_out

        return output

    def _lora_forward_tp_rowwise(self, x):
        """lora forward tp rowwise."""

        lora_input = self._make_packed_lora_input(x)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        out_size = self.base_layer.weight.size(0) // world_size
        if not lora_input.is_decoding:
            xa = mbgmm_a(lora_input.x,
                         lora_input.a_cache,
                         b_start_loc=lora_input.b_start_loc,
                         b_seq_lens=lora_input.b_seq_lens,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_seq_len=lora_input.max_seq_len,
                         max_rank=lora_input.max_rank)
            lora_out = mbgmm_b(xa,
                               lora_input.b_cache,
                               b_start_loc=lora_input.b_start_loc,
                               b_seq_lens=lora_input.b_seq_lens,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_seq_len=lora_input.max_seq_len,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)
        else:
            xa = mbgmv_a(lora_input.x,
                         lora_input.a_cache,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_rank=lora_input.max_rank)
            lora_out = mbgmv_b(xa,
                               lora_input.b_cache,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)

        base_out = self.base_layer(x)
        out_shape = base_out.shape
        base_out = base_out.flatten(0, -2)

        slice_start = rank * out_size
        slice_end = slice_start + out_size
        base_out[:, slice_start:slice_end] += lora_out
        base_out = base_out.reshape(out_shape)

        return base_out

    def _lora_forward_tp_colwise(self, x):
        """lora forward tp colwise."""

        def __gather_xa(xa):
            """gather xa."""
            gathered_xa = xa.new_empty(world_size, xa.size(0), xa.size(1))
            dist.all_gather_into_tensor(gathered_xa, xa)
            # TODO: gather would failed when adapters have different ranks.
            gathered_xa = gathered_xa.permute(1, 0, 2).flatten(-2, -1)
            return gathered_xa

        lora_input = self._make_packed_lora_input(x)
        world_size = dist.get_world_size()
        out_size = self.base_layer.weight.size(0)
        if not lora_input.is_decoding:
            xa = mbgmm_a(lora_input.x,
                         lora_input.a_cache,
                         b_start_loc=lora_input.b_start_loc,
                         b_seq_lens=lora_input.b_seq_lens,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_seq_len=lora_input.max_seq_len,
                         max_rank=lora_input.max_rank,
                         rank_step=world_size)
            gathered_xa = __gather_xa(xa)
            if len(lora_input.ranks) > 1:
                gathered_xa = rearange_all_gather(
                    gathered_xa,
                    b_start_loc=lora_input.b_start_loc,
                    b_seq_lens=lora_input.b_seq_lens,
                    adapter_ids=lora_input.b_adapter_ids,
                    ranks=lora_input.ranks,
                    world_size=world_size,
                    max_seq_len=lora_input.max_seq_len,
                    output=gathered_xa)
            lora_out = mbgmm_b(gathered_xa,
                               lora_input.b_cache,
                               b_start_loc=lora_input.b_start_loc,
                               b_seq_lens=lora_input.b_seq_lens,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_seq_len=lora_input.max_seq_len,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)
        else:
            xa = mbgmv_a(lora_input.x,
                         lora_input.a_cache,
                         b_adapter_ids=lora_input.b_adapter_ids,
                         rank_page_table=lora_input.rank_page_table,
                         rank_page_start=lora_input.rank_page_start,
                         ranks=lora_input.ranks,
                         max_rank=lora_input.max_rank,
                         rank_step=world_size)
            gathered_xa = __gather_xa(xa)
            if len(lora_input.ranks) > 1:
                gathered_xa = rearange_all_gather(
                    gathered_xa,
                    b_start_loc=lora_input.b_start_loc,
                    b_seq_lens=lora_input.b_seq_lens,
                    adapter_ids=lora_input.b_adapter_ids,
                    ranks=lora_input.ranks,
                    world_size=world_size,
                    max_seq_len=lora_input.max_seq_len,
                    output=gathered_xa)
            lora_out = mbgmv_b(gathered_xa,
                               lora_input.b_cache,
                               b_adapter_ids=lora_input.b_adapter_ids,
                               b_scaling=lora_input.b_scaling,
                               rank_page_table=lora_input.rank_page_table,
                               rank_page_start=lora_input.rank_page_start,
                               ranks=lora_input.ranks,
                               max_rank=lora_input.max_rank,
                               out_size=out_size)

        base_out = self.base_layer(x)
        lora_out = lora_out.reshape(base_out.shape)
        output = base_out + lora_out

        return output

    def _lora_forward_tp(self, x):
        """lora forward tp."""
        tp_mode = getattr(self, '_tp_mode', None)
        if tp_mode == 'rowwise':
            return self._lora_forward_tp_rowwise(x)
        elif tp_mode == 'colwise':
            return self._lora_forward_tp_colwise(x)
        else:
            assert tp_mode is None, 'tp_mode == None failed.'
            return self._lora_forward_local(x)

    def _lora_forward(self, x):
        """lora forward."""
        if dist.is_initialized():
            return self._lora_forward_tp(x)
        else:
            return self._lora_forward_local(x)

    def forward(self, x):
        """forward."""
        context = self.context.context
        max_rank = context.max_rank

        if max_rank == 0:
            return self.origin_mod.forward(x)
        else:
            return self._lora_forward(x)
