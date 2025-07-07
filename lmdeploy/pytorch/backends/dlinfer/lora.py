# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.kernels.dlinfer.fused_lora import fused_lora
from lmdeploy.pytorch.model_inputs import StepContextManager

from ..lora import AdapterInfo, LoRABuilder, LoRAImpl


@dataclass
class PackedLoRAInput:
    """Packed lora input."""
    x: torch.Tensor
    q_start_loc: torch.Tensor
    q_seqlens: torch.Tensor
    adapter_ids: torch.Tensor
    max_seq_len: int
    is_decoding: bool


class DlinferLoRAImpl(LoRAImpl):
    """Triton lora implementation."""

    @staticmethod
    def _make_packed_lora_input(x, ctx_mgr):
        """Make PackedLoRAInput."""
        context = ctx_mgr.current_context()

        # adapter cache
        max_q_seq_length = x.numel() // x.size(-1)

        return PackedLoRAInput(x=x.flatten(0, -2).contiguous(),
                               q_start_loc=context.q_start_loc,
                               q_seqlens=context.q_seqlens,
                               adapter_ids=context.local_adapter_ids,
                               max_seq_len=max_q_seq_length,
                               is_decoding=context.is_decoding)

    def forward(self,
                x: torch.Tensor,
                lora_A: torch.Tensor,
                lora_B: torch.Tensor,
                base_output: torch.Tensor,
                adapter_info: AdapterInfo,
                ctx_mgr: StepContextManager,
                colwise: bool,
                is_tp: bool = True):
        """forward."""
        lora_input = self._make_packed_lora_input(x, ctx_mgr)

        return fused_lora(
            lora_input.x,
            lora_A,
            lora_B,
            scaling=adapter_info.scalings,
            rank_start=adapter_info.rank_offsets,
            ranks=adapter_info.ranks,
            seq_start=lora_input.q_start_loc,
            seq_lens=lora_input.q_seqlens,
            adapter_ids=lora_input.adapter_ids,
            max_rank=adapter_info.max_rank,
            max_seqlen=lora_input.max_seq_len,
            slice_start=adapter_info.base_slice.start,
            slice_stop=adapter_info.base_slice.stop,
            slice_step=adapter_info.base_slice.step,
            output=base_output,
        )


class DlinferLoRABuilder(LoRABuilder):
    """Dlinfer lora layer builder."""

    @staticmethod
    def build():
        """build."""
        return DlinferLoRAImpl()
