# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
from torch import Tensor


def fused_lora(input: Tensor, lora_a: Tensor, lora_b: Tensor, scaling: Tensor, rank_start: Tensor, ranks: Tensor,
               seq_start: Tensor, seq_lens: Tensor, adapter_ids: Tensor, max_rank: int, max_seqlen: int,
               slice_start: int, slice_stop: int, slice_step: Optional[int], output: Optional[Tensor]):
    """Fused lora."""
    return ext_ops.fused_lora(input, lora_a, lora_b, scaling, rank_start, ranks, seq_start, seq_lens, adapter_ids,
                              max_rank, max_seqlen, slice_start, slice_stop, slice_step, output)
