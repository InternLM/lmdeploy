# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.default.conceptlm import DefaultConceptLMRuntimeOpsImpl


def _make_ops() -> DefaultConceptLMRuntimeOpsImpl:
    config = SimpleNamespace(
        concept_chunk_size=4,
        concept_chunk_merge_method='meanpooling',
        concept_shift_feature=True,
    )
    return DefaultConceptLMRuntimeOpsImpl(config)


def _make_prefill_attn_metadata(q_seqlens: list[int]):
    q_seqlens_tensor = torch.tensor(q_seqlens, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(q_seqlens_tensor, dim=0, dtype=torch.int32), (1, 0))
    return SimpleNamespace(
        is_decoding=False,
        q_seqlens=q_seqlens_tensor,
        q_start_loc=cu_seqlens[:-1],
        cu_seqlens_q=cu_seqlens,
        kv_seqlens=q_seqlens_tensor,
        max_q_seqlen=max(q_seqlens),
    )


def test_concept_prefill_position_ids_use_compressed_timeline():
    """HLM RoPE uses concept indices, not token-start positions."""
    ops = _make_ops()
    metadata = _make_prefill_attn_metadata([9, 10])
    position_ids = torch.tensor(list(range(9)) + list(range(10)), dtype=torch.long)

    concept_metadata = ops.build_prefill_metadata(metadata, position_ids)

    assert concept_metadata.concept_q_seqlens.tolist() == [2, 2]
    assert concept_metadata.concept_position_ids.tolist() == [0, 1, 0, 1]


def test_concept_prefill_position_ids_preserve_absolute_prefill_offset():
    """Chunked prefill keeps absolute concept index within the request."""
    ops = _make_ops()
    metadata = _make_prefill_attn_metadata([9])
    position_ids = torch.arange(8, 17, dtype=torch.long)

    concept_metadata = ops.build_prefill_metadata(metadata, position_ids)

    assert concept_metadata.concept_q_seqlens.tolist() == [2]
    assert concept_metadata.concept_position_ids.tolist() == [2, 3]


def test_concept_decode_position_ids_use_current_concept_index():
    ops = _make_ops()

    position_ids = torch.tensor([3, 4, 7, 8], dtype=torch.long)

    assert ops.decode_concept_position_ids(position_ids).tolist() == [0, 0, 1, 1]
