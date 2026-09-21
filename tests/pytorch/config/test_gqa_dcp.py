# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.config import DistConfig, ModelConfig


@pytest.mark.parametrize('dcp_size', [1, 2])
@pytest.mark.parametrize('policy', list(QuantPolicy))
def test_gqa_dcp_executor_cache_policy(monkeypatch, dcp_size, policy):
    from lmdeploy.pytorch.engine import executor
    from lmdeploy.pytorch.engine.executor import mp_executor

    model = SimpleNamespace(use_flash_mla=False, mla_index_topk=None)
    monkeypatch.setattr(ModelConfig, 'from_pretrained', Mock(return_value=model))
    constructor = Mock()
    monkeypatch.setattr(mp_executor, 'MPExecutor', constructor)
    cache = SimpleNamespace(block_size=16, quant_policy=policy)
    misc = SimpleNamespace(hf_overrides=None, model_format=None, memdecode_config=None, empty_init=False)
    kwargs = dict(model_path='', cache_config=cache, backend_config=None,
                  dist_config=DistConfig(tp=2, dcp=dcp_size), misc_config=misc,
                  distributed_executor_backend='mp')
    if dcp_size > 1 and policy in (QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.TURBO_QUANT):
        with pytest.raises(ValueError, match='unquantized or per-tensor FP8'):
            executor.build_executor(**kwargs)
        constructor.assert_not_called()
    else:
        assert executor.build_executor(**kwargs) is constructor.return_value
        assert constructor.call_args.kwargs['cache_config'].quant_policy == policy


@pytest.mark.parametrize('num_heads', [32, 64])
@pytest.mark.parametrize(('tp', 'dcp', 'supported'), [(4, 2, False), (8, 2, True), (8, 4, False), (16, 4, True)])
def test_qwen_gqa_dcp_head_replication(num_heads, tp, dcp, supported):
    config = SimpleNamespace(
        architectures=['Qwen3MoeForCausalLM'], model_type='qwen3_moe', hidden_size=2048,
        num_hidden_layers=1, num_attention_heads=num_heads, num_key_value_heads=4,
        head_dim=128, bos_token_id=1, eos_token_id=2, vocab_size=151936)
    if not supported:
        with pytest.raises(AssertionError, match='share replicated KV heads'):
            ModelConfig.from_hf_config(config, dist_config=DistConfig(tp=tp, dcp=dcp))
    else:
        model = ModelConfig.from_hf_config(config, dist_config=DistConfig(tp=tp, dcp=dcp))
        assert model.get_num_qkv_head_by_tp() == (num_heads // tp, 1)
        assert model.num_replicate_key_value_heads == tp // 4


@pytest.mark.parametrize('fa3', [True, False])
def test_gqa_dcp_dispatch(monkeypatch, fa3):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
    from lmdeploy.pytorch.backends.cuda import attention
    from lmdeploy.pytorch.backends.cuda.attention import cp
    constructor = Mock()
    monkeypatch.setattr(cp, 'DCPAttentionImpl', constructor)
    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (2, 0))
    monkeypatch.setattr(attention, '_enable_fa3', lambda *args: fa3)
    spec = PagedAttentionBuildSpec(num_heads=4, head_dim=128, num_kv_heads=1, v_head_dim=128,
                                   scale=None, alibi=False, sliding_window=None, logit_softcapping=0.0,
                                   causal=True, use_flash_mla=False, mla_index_topk=None,
                                   learnable_sink=False, block_sparse_size=1)
    assert attention._build_paged_attention(spec) is constructor.return_value
    assert constructor.call_args.kwargs['use_fa3'] == fa3


def test_gqa_prefix_workspace_accounts_for_separate_kv(monkeypatch):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends import cp_utils
    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (2, 0))
    planner = Mock(return_value=())
    monkeypatch.setattr(cp_utils, 'build_dcp_prefix_chunks', planner)
    model = ModelConfig(hidden_size=2048, num_layers=1, num_attention_heads=32,
                         num_key_value_heads=8, head_dim=128, k_head_dim=128, v_head_dim=128,
                         bos_token_id=1, eos_token_id=[2], dist_config=DistConfig(tp=8, dcp=2))
    metadata = SimpleNamespace(kv_seqlens=torch.tensor([65]), q_seqlens=torch.tensor([1]),
                                is_decoding=False, kv_flatten_size=65, max_kv_seqlen=65)
    step = SimpleNamespace(input_ids=torch.tensor([[1]]), model_config=model,
                           cache_config=SimpleNamespace(block_size=16))
    cp_utils.update_dcp_metadata(metadata, step)
    assert planner.call_args.kwargs['kv_width'] == 256
