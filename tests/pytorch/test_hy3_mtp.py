# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config

from lmdeploy.pytorch.backends import OpType
from lmdeploy.pytorch.backends.default import DefaultOpsBackend
from lmdeploy.pytorch.config import (
    CacheConfig,
    DistConfig,
    ModelConfig,
    QuantizationConfig,
)
from lmdeploy.pytorch.configurations.hy3 import Hy3ModelConfigBuilder
from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.model_inputs import BuildModelContext, ModelInputs
from lmdeploy.pytorch.models.hy3_mtp import HYV3MultiTokenPredictor
from lmdeploy.pytorch.models.patch import build_model_from_hf_config
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.spec_decode.proposers.base import build_specdecode_proposer
from lmdeploy.pytorch.spec_decode.proposers.deepseek_mtp import DeepseekMTP
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs


class _NoopBuilder:
    @staticmethod
    def build(*args, **kwargs):
        return torch.nn.Identity()


class _TestOpsBackend(DefaultOpsBackend):
    @classmethod
    def get_layer_impl_builder(cls, layer_type):
        unsupported_without_cuda = (
            OpType.PagedAttention,
            OpType.FusedMoE,
            OpType.LinearStaticF8,
            OpType.FusedMoEStaticF8,
        )
        if layer_type in unsupported_without_cuda:
            return _NoopBuilder
        return super().get_layer_impl_builder(layer_type)


def _patch_backend(monkeypatch):
    monkeypatch.setattr('lmdeploy.pytorch.backends.selector._get_backend', lambda: _TestOpsBackend)


def _make_config():
    return HYV3Config(
        architectures=['HYV3MTP'],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
        router_scaling_factor=2.826,
        num_nextn_predict_layers=1,
    )


def test_hy3_mtp_loads_bf16_checkpoint_weights(monkeypatch):
    _patch_backend(monkeypatch)
    config = _make_config()
    model = build_model_from_hf_config(config, dtype=torch.float32, device=torch.device('cpu'))
    eh_proj = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    q_proj = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    expert_gate = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    final_norm = torch.arange(16, dtype=torch.float32)
    model.load_weights(
        [
            ('model.layers.2.eh_proj.weight', eh_proj),
            ('model.layers.2.self_attn.q_proj.weight', q_proj),
            ('model.layers.2.mlp.experts.1.gate_proj.weight', expert_gate),
            ('model.layers.2.final_layernorm.weight', final_norm),
            ('model.layers.0.self_attn.q_proj.weight', torch.empty(1)),
        ]
    )
    layer = model.model.layers['2']
    assert model.get_input_embeddings() is None
    torch.testing.assert_close(layer.eh_proj.weight, eh_proj)
    torch.testing.assert_close(layer.mtp_block.self_attn.qkv_proj.weight[:16], q_proj)
    torch.testing.assert_close(layer.mtp_block.mlp.experts.gate_up.weight[1, :8], expert_gate)
    torch.testing.assert_close(layer.final_layernorm.weight, final_norm)


def test_hy3_mtp_skips_layer_local_shared_weights(monkeypatch):
    _patch_backend(monkeypatch)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    model.load_weights(
        [
            ('model.layers.2.embed_tokens.weight', torch.empty(1)),
            ('model.layers.2.shared_head.weight', torch.empty(1)),
        ]
    )


def test_hy3_mtp_wrapper_matches_vllm_reference_forward_formula(monkeypatch):
    _patch_backend(monkeypatch)
    torch.manual_seed(7)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    layer = model.model.layers['2']

    class _ReferenceBlock(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states * 0.25, hidden_states)

    layer.mtp_block = _ReferenceBlock()
    with torch.no_grad():
        layer.eh_proj.weight.copy_(torch.randn_like(layer.eh_proj.weight) * 0.02)
    input_embeddings = torch.randn(1, 3, 16)
    previous_hidden_states = torch.randn(1, 3, 16)
    rotary_pos_emb = (torch.empty(0), torch.empty(0))
    output = layer(
        input_embeddings,
        previous_hidden_states,
        rotary_pos_emb=rotary_pos_emb,
        past_key_value=[],
    )
    normalized_embeddings = torch.nn.functional.rms_norm(
        input_embeddings, (16,), layer.enorm.weight, model.config.rms_norm_eps
    )
    normalized_hidden_states = torch.nn.functional.rms_norm(
        previous_hidden_states, (16,), layer.hnorm.weight, model.config.rms_norm_eps
    )
    fused_hidden_states = torch.nn.functional.linear(
        torch.cat([normalized_embeddings, normalized_hidden_states], dim=-1),
        layer.eh_proj.weight,
    )
    expected = torch.nn.functional.rms_norm(
        fused_hidden_states * 1.25,
        (16,),
        layer.final_layernorm.weight,
        model.config.rms_norm_eps,
    )
    torch.testing.assert_close(output, expected)


def test_hy3_mtp_masks_position_zero_embedding(monkeypatch):
    _patch_backend(monkeypatch)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    embedding = torch.nn.Embedding(32, 16)
    model.set_input_embeddings(embedding)

    class _CaptureLayer(torch.nn.Module):
        def forward(self, input_embeddings, previous_hidden_states, **kwargs):
            self.input_embeddings = input_embeddings
            return previous_hidden_states

    capture_layer = _CaptureLayer()
    model.model.layers['2'] = capture_layer
    input_ids = torch.tensor([[3, 4]])
    position_ids = torch.tensor([[0, 1]])
    previous_hidden_states = torch.randn(1, 2, 16)
    provided_embeddings = embedding(input_ids)
    original_embeddings = provided_embeddings.detach().clone()
    output = model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        previous_hidden_states=previous_hidden_states,
        past_key_values=[[torch.empty(0), torch.empty(0)]],
        inputs_embeds=provided_embeddings,
    )
    torch.testing.assert_close(capture_layer.input_embeddings[:, 0], torch.zeros(1, 16))
    torch.testing.assert_close(capture_layer.input_embeddings[:, 1], original_embeddings[:, 1])
    torch.testing.assert_close(provided_embeddings, original_embeddings)
    assert output is previous_hidden_states


def test_hy3_mtp_prepares_speculative_runtime_inputs(monkeypatch):
    _patch_backend(monkeypatch)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    input_ids = torch.tensor([[3, 4]])
    position_ids = torch.tensor([[7, 8]])
    hidden_states = torch.randn(1, 2, 16)
    shifted_embeddings = torch.randn(1, 2, 16)
    attn_metadata = object()
    caches = [object()]
    context = SimpleNamespace(
        input_ids=input_ids,
        position_ids=position_ids,
        target_hidden_states=hidden_states,
        target_inputs_embeds=shifted_embeddings,
        attn_metadata=attn_metadata,
    )
    inputs = model.prepare_inputs_for_generation(past_key_values=caches, context=context)
    assert inputs['input_ids'] is input_ids
    assert inputs['position_ids'] is position_ids
    assert inputs['target_hidden_states'] is hidden_states
    assert inputs['inputs_embeds'] is shifted_embeddings
    assert inputs['attn_metadata'] is attn_metadata
    assert inputs['past_key_values'] is caches


def test_hy3_mtp_cuda_graph_buffers_include_target_hidden_states(monkeypatch):
    _patch_backend(monkeypatch)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    graph_meta = CudaGraphMeta(
        max_batchs=2,
        max_tokens=4,
        num_blocks=2,
        is_decoding=1,
        device=torch.device('cpu'),
        vocab_size=32,
    )
    input_ids = torch.tensor([[3, 4]])
    position_ids = torch.tensor([[7, 8]])
    attn_metadata = SimpleNamespace(
        block_offsets=torch.zeros((1, 1), dtype=torch.int32),
        q_start_loc=torch.zeros(1, dtype=torch.int32),
        q_seqlens=torch.tensor([2], dtype=torch.int32),
        kv_seqlens=torch.tensor([9], dtype=torch.int32),
    )
    target_hidden_states = torch.randn(1, 2, 16)
    common_inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'past_key_values': [],
        'attn_metadata': attn_metadata,
        'inputs_embeds': None,
        'target_hidden_states': target_hidden_states,
    }
    graph_meta.input_buffers = model.make_buffers_cudagraph(graph_meta, **common_inputs)
    new_inputs = model.fill_buffers_cudagraph(graph_meta, **common_inputs)
    target_buffer = graph_meta.input_buffers['target_hidden_states']
    assert target_buffer.shape == (1, 4, 16)
    torch.testing.assert_close(target_buffer[:, :2], target_hidden_states)
    assert new_inputs['target_hidden_states'] is target_buffer


def test_hy3_mtp_proposer_is_registered():
    config = SimpleNamespace(method='hy3_mtp', num_speculative_tokens=2)
    proposer = build_specdecode_proposer(config, device='cpu')
    assert type(proposer).__name__ == 'Hy3MTP'


def test_hy3_mtp_proposer_shares_target_embeddings(monkeypatch):
    proposer = build_specdecode_proposer(SimpleNamespace(method='hy3_mtp', num_speculative_tokens=2), device='cpu')
    shared_embedding = torch.nn.Embedding(32, 16)

    class _DraftModel:
        def set_input_embeddings(self, embedding):
            self.embedding = embedding

    draft_model = _DraftModel()

    def _build_model(self, empty_init, target_model=None, build_model_ctx=None):
        self.model = draft_model
        self.target_model = target_model

    monkeypatch.setattr(DeepseekMTP, 'build_model', _build_model)
    target_model = SimpleNamespace(get_input_embeddings=lambda: shared_embedding)
    proposer.build_model(empty_init=True, target_model=target_model)
    assert draft_model.embedding is shared_embedding


def test_hy3_mtp_proposer_uses_target_lm_head(monkeypatch):
    _patch_backend(monkeypatch)
    draft_model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    proposer = build_specdecode_proposer(SimpleNamespace(method='hy3_mtp', num_speculative_tokens=2), device='cpu')
    hidden_states = torch.randn(1, 2, 16)
    expected_logits = torch.randn(1, 2, 32)
    target_model = SimpleNamespace(get_logits=lambda states: expected_logits if states is hidden_states else None)
    proposer.model = draft_model
    proposer.target_model = target_model
    logits = proposer.get_logits(hidden_states)
    assert logits is expected_logits
    assert not hasattr(draft_model, 'get_logits')


def test_hy3_mtp_selects_only_checkpoint_mtp_layers(monkeypatch):
    _patch_backend(monkeypatch)
    model = build_model_from_hf_config(_make_config(), dtype=torch.float32, device=torch.device('cpu'))
    assert model.get_checkpoint_weight_prefixes() == ('model.layers.2.',)


def test_hy3_mtp_next_draft_step_advances_position_and_kv_length():
    proposer = build_specdecode_proposer(SimpleNamespace(method='hy3_mtp', num_speculative_tokens=2), device='cpu')
    inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12]]),
        seq_length=torch.tensor([3]),
        history_lengths=torch.tensor([5]),
        block_offsets=torch.zeros((1, 1), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=8,
        sum_kv_seqlen=8,
    )
    extra_inputs = ARSpecExtraInputs(last_token_indices=torch.tensor([2]))
    next_hidden_states = torch.randn(1, 1, 16)
    next_inputs = proposer.update_inputs_decoding(
        inputs,
        extra_inputs,
        next_input_ids=torch.tensor([[13]]),
        target_hidden_states=next_hidden_states,
        model_metas=[{'step': 1}],
    )
    torch.testing.assert_close(next_inputs.history_lengths, torch.tensor([8]))
    torch.testing.assert_close(next_inputs.seq_length, torch.tensor([1]))
    torch.testing.assert_close(next_inputs.target_position_ids, torch.tensor([[8]]))
    assert next_inputs.max_q_seqlen == 1
    assert next_inputs.max_kv_seqlen == 9
    assert next_inputs.sum_kv_seqlen == 9
    assert next_inputs.target_hidden_states is next_hidden_states


@torch.inference_mode()
def test_hy3_mtp_kv_cache_shapes_for_tp1_tp4_tp8(monkeypatch):
    _patch_backend(monkeypatch)
    cache_config = CacheConfig(
        max_batches=1,
        block_size=16,
        num_cpu_blocks=0,
        num_gpu_blocks=2,
        device_type='cuda',
    )
    for tp in (1, 4, 8):
        config = HYV3Config(
            architectures=['HYV3ForCausalLM'],
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=2,
            max_position_embeddings=128,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=16,
            first_k_dense_replace=1,
            router_scaling_factor=2.826,
            num_nextn_predict_layers=1,
        )
        model_config = ModelConfig.from_hf_config(
            config,
            dtype='bfloat16',
            dist_config=DistConfig(tp=tp),
            is_draft_model=True,
            spec_method='hy3_mtp',
            device_type='cuda',
        )
        (_, caches) = CacheEngine.allocate_caches(
            num_blocks=2,
            model_config=model_config,
            cache_config=cache_config,
            world_size=tp,
            device='meta',
        )
        (key_cache, value_cache) = caches[:2]
        assert model_config.num_layers == 1
        assert model_config.get_num_qkv_head_by_tp() == (64 // tp, 8 // tp)
        assert key_cache.shape == (1, 2, 16, 8 // tp, 2)
        assert value_cache.shape == (1, 2, 16, 8 // tp, 2)


def test_hy3_mtp_parameter_shapes_for_tp1_tp4_tp8(monkeypatch):
    _patch_backend(monkeypatch)
    for tp in (1, 4, 8):
        dist_config = DistConfig(tp=tp)
        rank_zero = DistGroup(rank=0)
        dist_context = DistContext(
            rank=0,
            dist_config=dist_config,
            tp_group=rank_zero,
            attn_tp_group=rank_zero,
            mlp_tp_group=rank_zero,
            moe_tp_group=rank_zero,
        )
        config = HYV3Config(
            architectures=['HYV3MTP'],
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=2,
            max_position_embeddings=128,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=16,
            first_k_dense_replace=1,
            router_scaling_factor=2.826,
            num_nextn_predict_layers=1,
        )
        with get_dist_manager().context(dist_context):
            model = build_model_from_hf_config(config, dtype=torch.bfloat16, device=torch.device('meta'))
        layer = model.model.layers['80']
        attention = layer.mtp_block.self_attn
        moe = layer.mtp_block.mlp
        assert layer.eh_proj.weight.shape == (128, 256)
        assert attention.qkv_proj.weight.shape == (160 // tp, 128)
        assert attention.o_proj.weight.shape == (128, 128 // tp)
        assert moe.router.gate.weight.shape == (8, 128)
        assert moe.expert_bias.shape == (8,)
        assert moe.experts.gate_up.weight.shape == (8, 32 // tp, 128)
        assert moe.experts.down.weight.shape == (8, 128, 16 // tp)
        assert moe.shared_mlp.gate_up_proj.weight.shape == (32 // tp, 128)
        assert moe.shared_mlp.down_proj.weight.shape == (128, 16 // tp)


def test_hy3_mtp_requires_checkpoint_layer():
    config = SimpleNamespace(model_type='hy_v3', num_hidden_layers=80, num_nextn_predict_layers=0)
    with pytest.raises(ValueError, match='at least one checkpoint MTP layer'):
        Hy3ModelConfigBuilder.build(config, is_draft_model=True, spec_method='hy3_mtp')
    with pytest.raises(ValueError, match='at least one checkpoint MTP layer'):
        HYV3MultiTokenPredictor(config)


def test_hy3_rejects_unsupported_speculative_method():
    config = SimpleNamespace(model_type='hy_v3', num_nextn_predict_layers=1)
    with pytest.raises(ValueError, match='Unsupported speculative method'):
        Hy3ModelConfigBuilder.build(config, spec_method='unsupported_method')


@pytest.mark.parametrize('reverse', [False, True])
def test_hy3_mtp_preserves_per_expert_input_scales(monkeypatch, reverse):
    _patch_backend(monkeypatch)
    config = _make_config()
    config.quantization_config = {
        'quant_method': 'fp8',
        'activation_scheme': 'static',
        'ignored_layers': ['lm_head', 'model.embed_tokens'],
    }
    build_context = BuildModelContext(quant_config=QuantizationConfig.from_config(config))
    model = build_model_from_hf_config(
        config, dtype=torch.bfloat16, device=torch.device('cpu'), build_model_ctx=build_context
    )
    prefix = 'model.layers.2.mlp.experts'
    weights = [
        (f'{prefix}.0.gate_proj.input_scale', torch.tensor([0.25])),
        (f'{prefix}.0.up_proj.input_scale', torch.tensor([0.25])),
        (f'{prefix}.1.gate_proj.input_scale', torch.tensor([0.5])),
        (f'{prefix}.1.up_proj.input_scale', torch.tensor([0.5])),
        (f'{prefix}.2.gate_proj.input_scale', torch.tensor([0.75])),
        (f'{prefix}.2.up_proj.input_scale', torch.tensor([0.75])),
        (f'{prefix}.3.gate_proj.input_scale', torch.tensor([1.0])),
        (f'{prefix}.3.up_proj.input_scale', torch.tensor([1.0])),
        (f'{prefix}.0.down_proj.input_scale', torch.tensor([0.4])),
        (f'{prefix}.1.down_proj.input_scale', torch.tensor([0.9])),
        (f'{prefix}.2.down_proj.input_scale', torch.tensor([0.6])),
        (f'{prefix}.3.down_proj.input_scale', torch.tensor([0.7])),
    ]
    if reverse:
        weights.reverse()
    model.load_weights(weights)
    experts = model.model.layers['2'].mtp_block.mlp.experts
    for linear_weights in (experts.gate_up, experts.down):
        linear_weights.update_weight(linear_weights.weight, linear_weights.weight_scale, linear_weights.input_scale)
    torch.testing.assert_close(experts.gate_up.input_scale, torch.tensor([0.25, 0.5, 0.75, 1.0]))
    torch.testing.assert_close(experts.down.input_scale, torch.tensor([0.4, 0.9, 0.6, 0.7]))


def test_hy3_mtp_compacts_and_reexpands_expert_input_scales(monkeypatch):
    _patch_backend(monkeypatch)
    config = _make_config()
    config.quantization_config = {
        'quant_method': 'fp8',
        'activation_scheme': 'static',
        'ignored_layers': ['lm_head', 'model.embed_tokens'],
    }
    build_context = BuildModelContext(quant_config=QuantizationConfig.from_config(config))
    model = build_model_from_hf_config(
        config, dtype=torch.bfloat16, device=torch.device('cpu'), build_model_ctx=build_context
    )
    prefix = 'model.layers.2.mlp.experts'
    model.load_weights(
        [
            (f'{prefix}.0.gate_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.0.up_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.1.gate_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.1.up_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.2.gate_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.2.up_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.3.gate_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.3.up_proj.input_scale', torch.tensor([0.25])),
            (f'{prefix}.0.down_proj.input_scale', torch.tensor([0.4])),
            (f'{prefix}.1.down_proj.input_scale', torch.tensor([0.4])),
            (f'{prefix}.2.down_proj.input_scale', torch.tensor([0.4])),
            (f'{prefix}.3.down_proj.input_scale', torch.tensor([0.4])),
        ]
    )
    experts = model.model.layers['2'].mtp_block.mlp.experts
    for linear_weights in (experts.gate_up, experts.down):
        linear_weights.update_weight(linear_weights.weight, linear_weights.weight_scale, linear_weights.input_scale)
    torch.testing.assert_close(experts.gate_up.input_scale, torch.tensor([0.25]))
    torch.testing.assert_close(experts.down.input_scale, torch.tensor([0.4]))
    model.load_weights(
        [
            (f'{prefix}.2.gate_proj.input_scale', torch.tensor([0.5])),
            (f'{prefix}.2.up_proj.input_scale', torch.tensor([0.5])),
        ]
    )
    torch.testing.assert_close(experts.gate_up.input_scale, torch.tensor([0.25, 0.25, 0.5, 0.25]))
    experts.gate_up.update_weight(experts.gate_up.weight, experts.gate_up.weight_scale, experts.gate_up.input_scale)
    torch.testing.assert_close(experts.gate_up.input_scale, torch.tensor([0.25, 0.25, 0.5, 0.25]))


def test_hy3_mtp_rejects_conflicting_gate_up_input_scales(monkeypatch):
    _patch_backend(monkeypatch)
    config = _make_config()
    config.quantization_config = {
        'quant_method': 'fp8',
        'activation_scheme': 'static',
        'ignored_layers': ['lm_head', 'model.embed_tokens'],
    }
    build_context = BuildModelContext(quant_config=QuantizationConfig.from_config(config))
    model = build_model_from_hf_config(
        config, dtype=torch.bfloat16, device=torch.device('cpu'), build_model_ctx=build_context
    )
    prefix = 'model.layers.2.mlp.experts'
    with pytest.raises(ValueError, match='must share the same input scale'):
        model.load_weights(
            [
                (f'{prefix}.0.gate_proj.input_scale', torch.tensor([0.25])),
                (f'{prefix}.0.up_proj.input_scale', torch.tensor([0.75])),
            ]
        )
