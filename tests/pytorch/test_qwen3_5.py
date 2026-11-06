# Copyright (c) OpenMMLab. All rights reserved.
"""Regression test for the AWQ branch of the patched fused-QKV weight_loader.

lmdeploy/pytorch/models/qwen3_5.py:_patch_qkv_weight_loader overrides weight_loader with a
non-uniform [key_dim, key_dim, value_dim] TP split. For AWQ modules it shards each section
with a plain Tensor.chunk, while the generic AWQ loaders in
lmdeploy/pytorch/nn/linear/awq.py use chunk_aligned, which additionally aligns chunk
boundaries to the int32 packing factor (elem_per_int) so a shard never splits a packed
value across ranks.

For every checkpoint layout this branch was written against (see #4899: key_dim=512,
value_dim=1024, in_features=2048, 4-bit AWQ), each packed section size is an exact
multiple of world_size * elem_per_int, so chunk_aligned has no remainder to redistribute
and produces the same shards as plain chunk. This test pins that equivalence for the
reported configuration, using chunk_aligned itself as the reference, so a future
configuration where it stops holding fails loudly instead of silently mis-sharding.
"""

import types

import pytest
import torch

from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5GatedDeltaNet
from lmdeploy.pytorch.nn.utils import chunk_aligned

# Values from the checkpoint that triggered #4899.
KEY_DIM = 512
VALUE_DIM = 1024
IN_FEATURES = 2048
W_BIT = 4
GROUP_SIZE = 128
ELEM_PER_INT = 32 // W_BIT
SECTIONS = (KEY_DIM, KEY_DIM, VALUE_DIM)
TOTAL_OUT = sum(SECTIONS)


class _StubAwqLinear:
    """Exposes only the attributes qkv_weight_loader reads from an AwqLinear."""

    def __init__(self, world_size: int, rank: int):
        self.is_tp = True
        self.elem_per_int = ELEM_PER_INT
        self._world_size = world_size
        self._rank = rank

    def get_tp_world_rank(self):
        return self._world_size, self._rank

    def setup_loaders(self):
        # _patch_qkv_weight_loader re-runs this after patching; the stub has no
        # parameters that need to be rebound to it.
        pass


def _patched_loader(world_size: int, rank: int):
    """A weight_loader taken from the real Qwen3_5GatedDeltaNet.qkv_weight_loader."""
    gdn = types.SimpleNamespace(key_dim=KEY_DIM, value_dim=VALUE_DIM)
    mod = _StubAwqLinear(world_size, rank)
    Qwen3_5GatedDeltaNet._patch_qkv_weight_loader(gdn, mod)
    return mod.weight_loader


def _full_weight(weight_type: str) -> torch.Tensor:
    torch.manual_seed(0)
    if weight_type == "scales":
        return torch.rand(IN_FEATURES // GROUP_SIZE, TOTAL_OUT, dtype=torch.float16)
    # qweight / qzeros: packed along the last dim, elem_per_int values per int32
    return torch.randint(0, 2**31 - 1, (IN_FEATURES, TOTAL_OUT // ELEM_PER_INT), dtype=torch.int32)


def _section_chunks(full_weight: torch.Tensor, weight_type: str, world_size: int):
    """Per section: its own slice of full_weight, and that slice's chunk_aligned split
    into world_size rank pieces."""
    align = 1 if weight_type == "scales" else ELEM_PER_INT
    offset = 0
    result = []
    for section in SECTIONS:
        width = section if weight_type == "scales" else section // ELEM_PER_INT
        part = full_weight[..., offset : offset + width]
        result.append((part, chunk_aligned(part, world_size, -1, align)))
        offset += width
    return result


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("weight_type", ["qweight", "scales", "qzeros"])
def test_awq_qkv_shard_matches_chunk_aligned(weight_type, world_size):
    full_weight = _full_weight(weight_type)
    sections = _section_chunks(full_weight, weight_type, world_size)

    for rank in range(world_size):
        # A rank's local weight is each section's rank-th chunk, concatenated in
        # section order -- the layout the local (post-TP) module expects.
        expected = torch.cat([chunks[rank] for _, chunks in sections], dim=-1)
        param = torch.nn.Parameter(torch.empty_like(expected), requires_grad=False)
        param._weight_type = weight_type

        _patched_loader(world_size, rank)(param, full_weight)

        torch.testing.assert_close(param.data, expected)

    # Every section is fully covered across ranks, with nothing dropped or
    # duplicated. A rank's shard already interleaves all three sections in its
    # own local layout, so coverage is checked per section rather than by
    # concatenating whole per-rank shards, which would not reconstruct
    # full_weight in its original column order.
    for section_slice, chunks in sections:
        torch.testing.assert_close(torch.cat(chunks, dim=-1), section_slice)
