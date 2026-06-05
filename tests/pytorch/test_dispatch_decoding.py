# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.model_inputs import BuildModelContext, DPMeta, StepContext, StepContextManager, step_ctx_manager


def _step_context(num_tokens: int, batch_size: int, is_decoding: bool = False, dp_meta=None):
    return StepContext(
        input_ids=torch.zeros((1, num_tokens), dtype=torch.long),
        model_config=None,
        cache_config=None,
        block_offsets=None,
        position_ids=None,
        attention_mask=None,
        q_seqlens=torch.ones(batch_size, dtype=torch.long),
        kv_seqlens=None,
        q_start_loc=None,
        kv_caches=None,
        is_decoding=is_decoding,
        sum_kv_seqlen=0,
        dp_meta=dp_meta,
    )


def _attention_metadata(batch_size: int, is_decoding: bool = False):
    return AttentionMetadata(
        is_decoding=is_decoding,
        block_offsets=None,
        q_seqlens=torch.ones(batch_size, dtype=torch.long),
    )


def test_dispatch_decoding_single_token_shape():
    mgr = StepContextManager(BuildModelContext(num_spec_tokens=0))
    with step_ctx_manager(mgr):
        batch_size = 4
        assert _step_context(num_tokens=batch_size, batch_size=batch_size).dispatch_decoding()
        assert _attention_metadata(batch_size=batch_size).dispatch_decoding(num_tokens=batch_size)


def test_dispatch_decoding_spec_shape():
    mgr = StepContextManager(BuildModelContext(num_spec_tokens=3))
    with step_ctx_manager(mgr):
        batch_size = 2
        num_tokens = batch_size * (1 + 3)
        assert _step_context(num_tokens=num_tokens, batch_size=batch_size).dispatch_decoding()
        assert _attention_metadata(batch_size=batch_size).dispatch_decoding(num_tokens=num_tokens)


def test_dispatch_decoding_shape_mismatch_is_prefill_shape():
    mgr = StepContextManager(BuildModelContext(num_spec_tokens=2))
    with step_ctx_manager(mgr):
        batch_size = 3
        num_tokens = batch_size * (1 + 2) - 1
        assert not _step_context(num_tokens=num_tokens, batch_size=batch_size).dispatch_decoding()
        assert not _attention_metadata(batch_size=batch_size).dispatch_decoding(num_tokens=num_tokens)


def test_dispatch_decoding_keeps_raw_decoding_true():
    mgr = StepContextManager(BuildModelContext(num_spec_tokens=4))
    with step_ctx_manager(mgr):
        batch_size = 2
        num_tokens = 1
        assert _step_context(num_tokens=num_tokens, batch_size=batch_size, is_decoding=True).dispatch_decoding()
        assert _attention_metadata(batch_size=batch_size, is_decoding=True).dispatch_decoding(num_tokens=num_tokens)


def test_dispatch_decoding_ignores_dp_meta():
    mgr = StepContextManager(BuildModelContext(num_spec_tokens=1))
    dp_meta = DPMeta(is_decoding=True, dp_is_decoding=True)
    with step_ctx_manager(mgr):
        assert not _step_context(num_tokens=3, batch_size=2, dp_meta=dp_meta).dispatch_decoding()
