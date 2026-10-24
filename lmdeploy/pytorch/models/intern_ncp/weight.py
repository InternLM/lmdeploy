# Copyright (c) OpenMMLab. All rights reserved.
import torch


def _repack_olmo_qkv_weight(loaded_weight: torch.Tensor, num_heads: int, head_dim: int):
    """Convert native OLMo per-head [Q,K,V] QKV packing to LMDeploy
    [Q][K][V]."""
    leading_shape = loaded_weight.shape[1:]
    loaded_weight = loaded_weight.reshape(num_heads, 3, head_dim, *leading_shape)
    query = loaded_weight[:, 0].flatten(0, 1)
    key = loaded_weight[:, 1].flatten(0, 1)
    value = loaded_weight[:, 2].flatten(0, 1)
    return torch.cat([query, key, value], dim=0)


def _load_stacked_codebook_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor, codebook_idx: int):
    """Load one native ``codebook.N`` checkpoint tensor into a stacked
    codebook."""
    assert 0 <= codebook_idx < param.size(0), f'Invalid codebook index: {codebook_idx}'
    target = param.data[codebook_idx]
    assert target.size() == loaded_weight.size(), (
        f'Attempted to load codebook weight ({loaded_weight.size()}) into parameter slice ({target.size()})')
    target.copy_(loaded_weight)
