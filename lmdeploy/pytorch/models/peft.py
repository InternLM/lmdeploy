# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..kernels.mbgmm import mbgmm_a, mbgmm_b
from ..kernels.mbgmv import mbgmv_a, mbgmv_b


class LoRALinear(torch.nn.Module):

    def _lora_forward(self, x):
        context = self.context.context

        # adapter inputs
        adapter_ids = context.adapter_ids
        adapter_offsets = context.adapter_offsets
        ranks = context.ranks
        max_rank = context.max_rank

        # adapter cache
        layer_idx = self.layer_idx
        k_cache, v_cache = context.kv_caches[layer_idx]
        cache_len = k_cache.size(0)
        a_cache = k_cache.view(cache_len, -1)
        b_cache = v_cache.view(cache_len, -1)

        # inputs meta
        q_start_loc = context.q_start_loc
        seq_length = context.seq_length
        max_seq_length = context.max_seq_length

        # mm or mv
        is_decoding = context.is_decoding

        x_shape = x.shape
        x = x.flatten(0, -2).contiguous()

        if layer_idx == 0:
            print(b_cache[145:169, :4])
        if not is_decoding:
            xa = mbgmm_a(x, a_cache,
                         b_start_loc=q_start_loc,
                         b_seq_lens=seq_length,
                         b_adapter_ids=adapter_ids,
                         rank_page_table=adapter_offsets,
                         ranks=ranks,
                         max_seq_len=max_seq_length,
                         max_rank=max_rank)
            lora_out = mbgmm_b(xa, b_cache,
                               b_start_loc=q_start_loc,
                               b_seq_lens=seq_length,
                               b_adapter_ids=adapter_ids,
                               rank_page_table=adapter_offsets,
                               ranks=ranks,
                               max_seq_len=max_seq_length,
                               max_rank=max_rank)
        else:
            xa = mbgmv_a(
                x, a_cache,
                b_adapter_ids=adapter_ids,
                rank_page_table=adapter_offsets,
                ranks=ranks,
                max_rank=max_rank)
            lora_out = mbgmv_b(xa, b_cache,
                               b_adapter_ids=adapter_ids,
                               rank_page_table=adapter_offsets,
                               ranks=ranks,
                               max_rank=max_rank)

        base_out = self.base_layer(x)
        output = base_out + lora_out
        output = output.unflatten(0, x_shape[:-1])

        return output

    def forward(self, x):
        context = self.context.context
        max_rank = context.max_rank

        if max_rank == 0:
            origin_mod = self.origin_mod
            return origin_mod.forward(x)
        else:
            return self._lora_forward(x)
