# Copyright (c) OpenMMLab. All rights reserved.
def div_up(a: int, b: int):
    """div up."""
    return (a + b - 1) // b


def get_distribute_size(feature_size: int,
                        world_size: int,
                        rank: int,
                        align: int = 1):
    """update feature size."""
    assert feature_size % align == 0
    aligned_size = feature_size // align
    align_per_rank = div_up(aligned_size, world_size)
    prev_feats = align_per_rank * rank
    updated_aligned_size = min(align_per_rank, aligned_size - prev_feats)
    return updated_aligned_size * align
