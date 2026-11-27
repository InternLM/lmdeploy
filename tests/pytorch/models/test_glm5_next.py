# Copyright (c) OpenMMLab. All rights reserved.
"""GLM target preparation must preserve the shared DP graph contract."""
from types import SimpleNamespace

import torch
from torch import nn

from lmdeploy.pytorch.backends.cuda.graph_runner.standard import StandardDecoderPiecewiseGraphRuntime
from lmdeploy.pytorch.consts import (
    GLM5_KDA_CONV_STATE,
    GLM5_KDA_RECURRENT_STATE,
    GLM5_KPOOL_TAIL_K_STATE,
    GLM5_KPOOL_TAIL_SCORE_STATE,
)
from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration


def test_mixed_decode_reuses_prefill_graph_key():
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=0)
    model.ctx_mgr = SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=5))
    state_names = (GLM5_KDA_CONV_STATE, GLM5_KDA_RECURRENT_STATE,
                   GLM5_KPOOL_TAIL_K_STATE, GLM5_KPOOL_TAIL_SCORE_STATE)
    context = SimpleNamespace(
        named_state_caches={name: torch.empty(1, 0) for name in state_names},
        input_ids=torch.zeros(1, 6, dtype=torch.long), position_ids=torch.arange(6)[None],
        input_multimodals=None, input_embeddings=None, input_embedding_indexing=None,
        attn_metadata=None, state_offsets=None, is_decoding=False, global_is_decoding=lambda: False,
        enable_microbatch=False, local_adapter_ids=None, is_dummy=False)
    runtime = StandardDecoderPiecewiseGraphRuntime(model, max_capture_tokens=6)
    prefill = model.prepare_inputs_for_generation([], context=context)
    key = runtime.get_piecewise_graph_descriptor(context, prefill)
    assert key is not None
    context.is_decoding = True
    mixed = model.prepare_inputs_for_generation([], context=context)
    assert runtime.get_piecewise_graph_descriptor(context, mixed) == key
    context.global_is_decoding = lambda: True
    decode = model.prepare_inputs_for_generation([], context=context)
    assert not decode['return_input_embeds']
