# Copyright (c) OpenMMLab. All rights reserved.
import torch


def div_up(a: int, b: int):
    """Div up."""
    return (a + b - 1) // b


def get_distribute_size(feature_size: int, world_size: int, rank: int, align: int = 1):
    """Update feature size."""
    assert feature_size % align == 0
    aligned_size = feature_size // align
    # try to make every rank has same amount of feats
    updated_aligned_size = aligned_size // world_size
    # if there are still some remain, given them to
    # each rank
    if rank < aligned_size % world_size:
        updated_aligned_size += 1
    return updated_aligned_size * align


def chunk_aligned(weight: torch.Tensor, chunks: int, dim: int, align: int):
    """Chunk aligned."""
    if align == 1:
        return weight.chunk(chunks, dim=dim)
    size = weight.size(dim)
    assert size % align == 0
    aligned_size = size // align

    # try best to evenly split chunks
    align_per_chunk = aligned_size // chunks
    remain = aligned_size % chunks
    sections = [align_per_chunk + int(c < remain) for c in range(chunks)]
    sections = [sec * align for sec in sections]
    return weight.split(sections, dim=dim)
