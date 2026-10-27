# Copyright (c) OpenMMLab. All rights reserved.
"""Model geometry and physical bindings; not compact serving integration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SSMStateCapacity, StateCacheSpec
from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder
from lmdeploy.pytorch.engine.cache_engine import state as state_module
from lmdeploy.pytorch.engine.cache_engine.schema import build_state_cache_tensor_specs
from lmdeploy.pytorch.engine.cache_engine.state import SSMStatePlan, StateCacheEngine


def _qwen(k=0, draft=False, tp=1, device='cuda', family='qwen3_5'):
    text = SimpleNamespace(hidden_size=512, num_hidden_layers=4, num_attention_heads=4,
                           num_key_value_heads=4, bos_token_id=1, eos_token_id=2, vocab_size=128,
                           layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
                           linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=2,
                           linear_num_value_heads=4, linear_conv_kernel_dim=4, mtp_num_hidden_layers=1)
    hf = SimpleNamespace(model_type=family, text_config=text, architectures=['Qwen3_5ForConditionalGeneration'])
    cfg = Qwen3_5ModelConfigBuilder.build(hf, num_spec_tokens=k, spec_method='qwen3_5_mtp' if k or draft else None,
                                         is_draft_model=draft, tp=tp, device_type=device)
    return cfg, hf


def _glm(monkeypatch, k=0, draft=False, tp=1, device='cuda'):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    hf = Glm5NextConfig()
    text = hf.text_config
    text.num_hidden_layers = 4
    text.layer_types = ['linear_attention', 'full_attention', 'linear_attention', 'full_attention']
    text.linear_num_heads, text.linear_head_dim, text.linear_conv_kernel_dim = 4, 128, 4
    cfg = Glm5NextModelConfigBuilder.build(hf, num_spec_tokens=k, is_draft_model=draft, tp=tp, device_type=device)
    return cfg, hf


@pytest.mark.parametrize('family', ['qwen', 'glm'])
@pytest.mark.parametrize('k', [0, 3, 5])
@pytest.mark.parametrize('tp', [1, 2])
def test_model_compact_shapes_use_canonical_and_auxiliary_geometry(monkeypatch, family, k, tp):
    model, _ = _qwen(k, tp=tp) if family == 'qwen' else _glm(monkeypatch, k, tp=tp)
    before = [(spec.name, spec.shape, spec.dtype) for spec in model.state_cache_specs]
    assert [(spec.shape, spec.dtype) for spec in model.state_cache_specs] == model.states_shapes
    capacity = SSMStateCapacity(4, k + 1, 2, 1)
    plan = SSMStatePlan.from_model_config(model, capacity)
    assert [(spec.name, spec.shape, spec.dtype) for spec in model.state_cache_specs] == before
    assert plan.unit_specs[0].desc.shape[-1] == 3
    assert len(plan.unit_specs[1].desc.shape) == 4  # L,HV,V,K, no speculative ring
    assert plan.unit_specs[1].desc.shape[1] == 4 // tp
    assert model.state_cache_specs[0].shape[-1] == 3
    assert len(model.state_cache_specs[1].shape) == 4
    if family == 'qwen':
        # Named and anonymous declarations describe the same canonical units.
        assert build_state_cache_tensor_specs(model.states_shapes) == build_state_cache_tensor_specs(
            (), model.state_cache_specs)
        assert not plan.auxiliary_specs
    else:
        assert len(plan.auxiliary_specs) == 2
        assert all(spec.desc.shape[1] == 5 + k for spec in plan.auxiliary_specs)
        assert all(not spec.store_cacheable for spec in plan.auxiliary_specs)
        assert plan.unit_specs[1].desc.dtype == torch.float32
    unit_bytes, aux_bytes = plan.get_nbytes()
    units, aux = plan.allocate('meta')
    assert (unit_bytes, aux_bytes) == (units.nbytes, aux.nbytes)
    assert units.tensor_views[0].shape[:2] == (capacity.num_units, 2)
    if family == 'glm':
        assert aux.tensor_views[0].shape[:2] == (capacity.request_slots, 2)


@pytest.mark.parametrize('family', ['qwen', 'glm'])
def test_draft_has_no_compact_resources_or_backend_lookup(monkeypatch, family):
    model, _ = _qwen(3, draft=True) if family == 'qwen' else _glm(monkeypatch, 3, draft=True)
    assert not model.state_cache_is_indexed
    assert not model.state_cache_specs and not model.states_shapes
    monkeypatch.setattr(state_module, 'get_backend', lambda: pytest.fail('Stateless plan must not select a backend'))
    plan = SSMStatePlan.from_model_config(model, SSMStateCapacity(0, 0))
    assert plan.get_nbytes() == (0, 0)
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)
    engine = StateCacheEngine(cache, model, compact_plan=plan)
    assert not engine.state_caches and not engine.connector_state_cache_pools
    assert engine.allocation.nbytes == 0


def test_unknown_model_and_backend_require_explicit_capability(monkeypatch):
    model, _ = _qwen()
    opaque = replace(model, state_cache_specs=[])
    with pytest.raises(ValueError, match='does not declare compact'):
        SSMStatePlan.from_model_config(opaque, SSMStateCapacity(2, 1))
    opaque = replace(model, state_cache_is_indexed=False,
                     state_cache_specs=[StateCacheSpec('opaque', (3, 5), torch.float32)])
    with pytest.raises(ValueError, match='does not declare indexed'):
        SSMStatePlan.from_model_config(opaque, SSMStateCapacity(2, 1))
    monkeypatch.setattr(state_module, 'get_backend', lambda: SimpleNamespace(get_name=lambda: 'unknown'))
    with pytest.raises(ValueError, match='requires the CUDA'):
        SSMStatePlan.from_model_config(model, SSMStateCapacity(2, 1))


@pytest.mark.parametrize('family', ['qwen', 'glm'])
def test_real_allocation_preserves_model_view_order_and_forbids_payload_conversion(monkeypatch, family):
    model, hf = _qwen(3) if family == 'qwen' else _glm(monkeypatch, 3)
    capacity = SSMStateCapacity(4, 4, 2, 1)
    plan = SSMStatePlan.from_model_config(model, capacity)
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0,
                        num_state_caches=7, states_shapes=model.states_shapes)
    # The compact allocation path must never allocate the legacy ring alongside it.
    monkeypatch.setattr(state_module, '_allocate_state_cache_groups',
                        lambda *a, **kw: pytest.fail('Compact mode must replace legacy allocation'))
    engine = StateCacheEngine(cache, model, compact_plan=plan)
    assert engine.allocation.nbytes == sum(plan.get_nbytes())
    assert len(engine.allocation.pools) == len(engine.unit_allocation.pools) + len(engine.auxiliary_allocation.pools)
    assert len(engine.connector_state_cache_pools) == len(engine.unit_allocation.pools)
    for spec, tensor in zip(plan.tensor_specs, engine.state_caches):
        assert engine.named_state_caches[spec.name] is tensor
    with pytest.raises(RuntimeError, match='read-only'):
        engine.zero_slots(torch.tensor([1], device='cuda'), torch.tensor([True], device='cuda'))
    with pytest.raises(RuntimeError, match='share units'):
        engine.copy_slots(1, 2, checkpoint_only=True)
    with pytest.raises(ValueError, match='does not match'):
        StateCacheEngine(cache, replace(model, state_cache_specs=[]), compact_plan=plan)

    # Exercise the real model's cache adapter, without constructing weights.
    context = SimpleNamespace(state_caches=engine.state_caches, named_state_caches=engine.named_state_caches,
                              input_ids=None, position_ids=None, attn_metadata=None, input_multimodals=None,
                              input_embeddings=None, input_embedding_indexing=None, state_offsets=None,
                              is_decoding=False, is_chunk_multimodal=False)
    full_kv = [(object(), object()), (object(), object())]
    if family == 'qwen':
        from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        fake_model = SimpleNamespace(config=hf, is_spec_decoding=True)
        outputs = Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation(fake_model, full_kv, context=context)
    else:
        from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration
        fake_model = SimpleNamespace(
            config=hf.text_config, ctx_mgr=SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=3)))
        outputs = Glm5NextForConditionalGeneration.prepare_inputs_for_generation(fake_model, full_kv, context=context)
    interleaved = outputs['past_key_values']
    assert len(interleaved) == 4
    for row, layer in enumerate((0, 2)):
        assert interleaved[layer][0].data_ptr() == engine.state_caches[0][:, row].data_ptr()
        assert interleaved[layer][1].data_ptr() == engine.state_caches[1][:, row].data_ptr()
        assert interleaved[layer][0].shape[0] == capacity.num_units
        assert interleaved[layer][0].shape[-1] == 3
        assert interleaved[layer][1].ndim == 4
    if family == 'glm':
        for row, layer in enumerate((1, 3)):
            assert interleaved[layer][:2] == full_kv[row]
            assert interleaved[layer][2].shape[0] == capacity.request_slots
            assert interleaved[layer][2].shape[1] == 8  # private MTP tail unchanged
            assert interleaved[layer][2].data_ptr() == engine.state_caches[2][:, row].data_ptr()


@pytest.mark.parametrize('draft', [False, True])
def test_glm52_target_and_draft_stay_stateless(monkeypatch, draft):
    from lmdeploy.pytorch.configurations.glm_moe_dsa import GlmMoeDsaModelConfigBuilder

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    hf = SimpleNamespace(model_type='glm_moe_dsa', architectures=['GlmMoeDsaForCausalLM'], hidden_size=512,
                         num_attention_heads=4, num_hidden_layers=4, num_nextn_predict_layers=1,
                         qk_head_dim=192, qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=128,
                         eos_token_id=2, vocab_size=128, index_topk=512)
    model = GlmMoeDsaModelConfigBuilder.build(hf, is_draft_model=draft, spec_method='deepseek_mtp', num_spec_tokens=3)
    assert not model.state_cache_is_indexed
    assert not model.state_cache_specs and not model.states_shapes
    monkeypatch.setattr(state_module, 'get_backend', lambda: pytest.fail('Stateless model must not build kernels'))
    plan = SSMStatePlan.from_model_config(model, SSMStateCapacity(0, 0))
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)
    engine = StateCacheEngine(cache, model, compact_plan=plan)
    assert engine.allocation.nbytes == 0
    assert not engine.state_caches and not engine.connector_state_cache_pools
    assert plan.capacity.num_units == plan.capacity.bank_width == plan.capacity.request_slots == 0


def test_engine_rejects_changed_compact_contract_before_allocation(monkeypatch):
    model, _ = _qwen()
    plan = SSMStatePlan.from_model_config(model, SSMStateCapacity(4, 1))
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0,
                        states_shapes=model.states_shapes)
    monkeypatch.setattr(plan, 'allocate', lambda *_: pytest.fail('Validate model bindings before allocating'))
    model.state_cache_specs[0].shape = (2, 1024, 2)
    with pytest.raises(ValueError, match='does not match'):
        StateCacheEngine(cache, model, compact_plan=plan)


def test_qwen_compiled_allocation_executes_indexed_prefill_without_adapter_copies():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn, indexed_causal_conv1d_fwd
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_gated_delta_rule

    torch.manual_seed(2048)
    model, hf = _qwen(3)
    plan = SSMStatePlan.from_model_config(model, SSMStateCapacity(4, 4, 2, 1))
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0,
                        num_state_caches=7, states_shapes=model.states_shapes)
    engine = StateCacheEngine(cache, model, compact_plan=plan)
    conv_cache, recurrent_cache = engine.state_caches
    conv_cache[1].normal_(std=0.1)
    recurrent_cache[1].normal_(std=0.1)
    before_conv, before_recurrent = conv_cache.clone(), recurrent_cache.clone()
    src, dst, cp = [torch.tensor([i], device='cuda') for i in (1, 2, 3)]
    cp_args = dict(checkpoint_ids=cp, checkpoint_offsets=torch.tensor([64], device='cuda'))
    seq = torch.zeros(72, device='cuda', dtype=torch.int32)
    cu = torch.tensor([0, 72], device='cuda')
    heads, value_heads, dim = hf.text_config.linear_num_key_heads, hf.text_config.linear_num_value_heads, 128
    for layer in range(2):
        conv = conv_cache[:, layer]
        recurrent = recurrent_cache[:, layer]
        channels = conv.shape[1]
        x = torch.randn(72, channels, device='cuda', dtype=conv.dtype).T.unsqueeze(0)
        weight = torch.randn(channels, 4, device='cuda', dtype=conv.dtype) / 8
        expected_conv = causal_conv1d_fn(x, weight, seq_idx=seq, initial_states=conv[1:2].contiguous())
        mixed = indexed_causal_conv1d_fwd(x, weight, conv, src, dst, seq_idx=seq, cu_seqlens=cu, **cp_args)
        torch.testing.assert_close(mixed, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(conv[3], x[0, :, 61:64], rtol=0, atol=0)
        torch.testing.assert_close(conv[2], x[0, :, -3:], rtol=0, atol=0)
        q, k, v = mixed.transpose(1, 2).split([heads * dim, heads * dim, value_heads * dim], dim=-1)
        q, k = [torch.nn.functional.normalize(t.reshape(1, 72, heads, dim).float(), dim=-1).to(conv.dtype)
                for t in (q, k)]
        v = v.reshape(1, 72, value_heads, dim).contiguous()
        g = -torch.rand(1, 72, value_heads, device='cuda') / 10
        beta = torch.rand_like(g)
        expected, expected_final = chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=recurrent[1:2].contiguous().float(),
            state_v_first=True, output_final_state=True)
        actual = indexed_chunk_gated_delta_rule(q, k, v, g, beta, recurrent, src, dst,
                                                state_v_first=True, **cp_args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(recurrent[2:3], expected_final.to(recurrent.dtype), rtol=0, atol=0)
    untouched = torch.ones(plan.capacity.num_units, device='cuda', dtype=torch.bool)
    untouched[2:4] = False
    torch.testing.assert_close(conv_cache[untouched], before_conv[untouched], rtol=0, atol=0)
    torch.testing.assert_close(recurrent_cache[untouched], before_recurrent[untouched], rtol=0, atol=0)


@pytest.mark.parametrize('family', ['qwen', 'glm', 'intern_s2_preview'])
@pytest.mark.parametrize('device', ['cuda', 'ascend'])
def test_indexed_contract_does_not_migrate_other_consumers(monkeypatch, family, device):
    model, _ = (_glm(monkeypatch, 5, device=device) if family == 'glm' else
                _qwen(5, device=device, family='qwen3_5' if family == 'qwen' else family))
    indexed = device == 'cuda' and family != 'intern_s2_preview'
    assert model.state_cache_is_indexed == indexed
    assert model.states_shapes[0][0][-1] == (3 if indexed else 9)
    assert len(model.states_shapes[1][0]) == (4 if indexed else 5)
    if family != 'glm' and not indexed:
        assert not model.state_cache_specs
