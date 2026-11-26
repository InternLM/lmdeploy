# Copyright (c) OpenMMLab. All rights reserved.
"""Construction-time coverage for unsupported DCP attention/cache layouts."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.pytorch.config import CacheConfig, DistConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


@pytest.mark.parametrize('window', [None, -1, (-1, -1), 16, (16, 0), (-1, 16)])
@pytest.mark.parametrize('dcp', [1, 2])
def test_effective_layer_window_guard(monkeypatch, window, dcp):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends.cuda import attention
    from lmdeploy.pytorch.backends.cuda.attention import cp

    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (dcp, 0))
    monkeypatch.setattr(attention, '_enable_fa3', lambda *args: False)
    dense = Mock()
    dcp_impl = Mock()
    monkeypatch.setattr(attention, 'TritonAttentionImpl', dense)
    monkeypatch.setattr(cp, 'DCPAttentionImpl', dcp_impl)
    spec = PagedAttentionBuildSpec(
        num_heads=4, head_dim=128, scale=None, num_kv_heads=1, v_head_dim=128,
        alibi=False, sliding_window=window, logit_softcapping=0.0, causal=True,
        use_flash_mla=False, mla_index_topk=None, learnable_sink=False, block_sparse_size=1)
    # A global layer can build even when the next local layer must fail. This
    # exercises mixed-window models whose model-wide window is disabled.
    attention._build_paged_attention(replace(spec, sliding_window=-1))
    if dcp > 1 and window not in (None, -1, (-1, -1)):
        with pytest.raises(ValueError, match=r'sliding_window=.*Use dcp=1'):
            attention._build_paged_attention(spec)
        assert dcp_impl.call_count == 1
    else:
        attention._build_paged_attention(spec)
        assert (dcp_impl if dcp > 1 else dense).call_count == 2


@pytest.mark.parametrize('dcp', [1, 2])
def test_noncausal_gqa_guard(monkeypatch, dcp):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends.cuda import attention

    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (dcp, 0))
    monkeypatch.setattr(attention, '_enable_fa3', lambda *args: False)
    dense = Mock()
    monkeypatch.setattr(attention, 'TritonAttentionImpl', dense)
    spec = PagedAttentionBuildSpec(
        num_heads=4, head_dim=128, scale=None, num_kv_heads=1, v_head_dim=128,
        alibi=False, sliding_window=None, logit_softcapping=0.0, causal=False,
        use_flash_mla=False, mla_index_topk=None, learnable_sink=False, block_sparse_size=1)
    if dcp > 1:
        with pytest.raises(ValueError, match=r'causal=False.*Use dcp=1'):
            attention._build_paged_attention(spec)
        dense.assert_not_called()
    else:
        attention._build_paged_attention(spec)
        assert dense.call_args.kwargs['causal'] is False


@pytest.mark.parametrize('dcp', [1, 2, 4, 8])
@pytest.mark.parametrize('method,arch', [
    ('dflash', 'DFlashDraftModel'),
    ('dspark', 'Qwen3DSparkModel'),
    ('eagle3', 'Eagle3LlamaForCausalLM'),
    ('eagle3', 'Eagle3DeepseekV2ForCausalLM'),
    ('deepseek_mtp', None),
])
def test_speculative_dcp_layout_guard(monkeypatch, tmp_path, dcp, method, arch):
    from lmdeploy.pytorch.spec_decode import config as module

    monkeypatch.setattr(module, 'config_from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(architectures=[arch]))
    build_draft = Mock()
    monkeypatch.setattr(module.SpecDecodeConfig, 'from_config', build_draft)
    dist_config = DistConfig(tp=8, dcp=dcp)
    incompatible = dcp > 1 and (method in ('dflash', 'dspark') or arch == 'Eagle3LlamaForCausalLM')
    kwargs = dict(target_model='target',
                  speculative_config=SpeculativeConfig(method=method, model=str(tmp_path), num_speculative_tokens=4),
                  engine_config=PytorchEngineConfig(tp=8, dcp=dcp),
                  cache_config=CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=8),
                  dist_config=dist_config)
    if incompatible:
        with pytest.raises(ValueError, match=r'(dflash|dspark|draft_arch=Eagle3LlamaForCausalLM).*dcp=1'):
            ConfigBuilder.build_specdecode_config(**kwargs)
        build_draft.assert_not_called()
    else:
        ConfigBuilder.build_specdecode_config(**kwargs)
        assert build_draft.call_args.kwargs['dist_config'].dcp == dcp
