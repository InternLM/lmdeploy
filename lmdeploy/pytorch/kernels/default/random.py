# Copyright (c) OpenMMLab. All rights reserved.
import torch


def rejection_uniform(seeds: torch.Tensor, positions: torch.Tensor, lanes: torch.Tensor, stream: int,
                      dtype: torch.dtype = torch.float64):
    """Philox4x32-10 uniforms keyed by request seed, token position, lane, and
    use.

    Inputs broadcast together. Streams 1 and 2 serve acceptance and recovery; stream 0 is reserved for ordinary AR
    sampling. Matches the CUDA sampler.
    """
    mask = 0xffffffff
    c0, c1 = lanes, positions & mask
    c2, c3 = positions >> 32, torch.full_like(positions, stream)
    k0, k1 = seeds & mask, (seeds >> 32) & mask
    for _ in range(10):
        product0, product2 = c0 * 0xd2511f53, c2 * 0xcd9e8d57
        c0, c1, c2, c3 = (
            ((product2 >> 32) & mask) ^ c1 ^ k0,
            product2 & mask,
            ((product0 >> 32) & mask) ^ c3 ^ k1,
            product0 & mask,
        )
        k0, k1 = (k0 + 0x9e3779b9) & mask, (k1 + 0xbb67ae85) & mask
    uniform = (c0.to(dtype) + 0.5) * 2.3283064365386963e-10
    return uniform.clamp_max_(1.0 - torch.finfo(dtype).eps / 2)
