# Copyright (c) OpenMMLab. All rights reserved.
import torch


def continuous_tensor(inputs: torch.Tensor, seq_length: torch.LongTensor):
    """Convert batched tensor to continuous tensor.

    Args:
        inputs (Tensor): batched tensor.
        seq_length (Tensor): length of each sequence.

    Return:
        Tensor: continuoused tensor.
    """
    assert inputs.dim() > 1
    if inputs.size(1) == 1:
        return inputs.reshape(1, -1)

    inputs = [inp[:slen] for inp, slen in zip(inputs, seq_length)]

    inputs = torch.cat(inputs).unsqueeze(0)
    return inputs


def batch_tensor(inputs: torch.Tensor, seq_length: torch.LongTensor):
    """Convert continuoused tensor to batched tensor.

    Args:
        inputs (Tensor): continuoused tensor.
        seq_length (Tensor): length of each sequence.

    Return:
        Tensor: batched tensor.
    """
    from torch.nn.utils.rnn import pad_sequence
    end_loc = seq_length.cumsum(0)
    start_loc = end_loc - seq_length

    inputs = [inputs[0, sloc:eloc] for sloc, eloc in zip(start_loc, end_loc)]
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs


def page_cache(paged_cache: torch.Tensor,
               batched_cache: torch.Tensor,
               cache_length: torch.Tensor,
               block_offsets: torch.Tensor,
               permute_head: bool = True):
    """Convert batched cache to paged cache.

    Args:
        paged_cache (Tensor): Output paged cache.
        batched_cache (Tensor): Input batched cache.
        cache_length (Tensor): length of the cache.
        block_offsets (Tensor): Offset of each blocks.
    """
    assert block_offsets.dim() == 2
    block_size = paged_cache.size(1)
    batch_size = batched_cache.size(0)
    if permute_head:
        batched_cache = batched_cache.permute(0, 2, 1, 3)

    for b_idx in range(batch_size):
        cache_len = cache_length[b_idx]
        b_cache = batched_cache[b_idx]
        block_off = block_offsets[b_idx]
        block_off_idx = 0
        for s_start in range(0, cache_len, block_size):
            s_end = min(s_start + block_size, cache_len)
            s_len = s_end - s_start
            b_off = block_off[block_off_idx]
            paged_cache[b_off, :s_len] = b_cache[s_start:s_end]
            block_off_idx += 1
