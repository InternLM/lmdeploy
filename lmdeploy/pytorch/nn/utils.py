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
    # try to make every rank has same amount of feats
    updated_aligned_size = aligned_size // world_size
    # if there are still some remain, given them to
    # each rank
    if rank < aligned_size % world_size:
        updated_aligned_size += 1
    return updated_aligned_size * align
