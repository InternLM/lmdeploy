from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.backends.default.apply_rotary_emb import DefaultApplyRotaryEmbImpl, rotate_interleaved
from lmdeploy.pytorch.models.deepseek_v2 import DeepseekV2BMM, DeepseekV2ForCausalLM, DeepseekV2MoE, MoEGate
from lmdeploy.pytorch.models.deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32ForCausalLM,
    DSATopKIndicesBuffer,
    Indexer,
    _dequantize_blocked_fp8,
    get_full_indexer_layer_ids,
    get_layer_indexer_type,
)


def _patch_minimal_attention(attn, monkeypatch, dp=1, attn_tp=1, seen=None):
    """Patch a DSA attention module down to the top-k routing contract."""

    def fake_qkv_proj(hidden_states, num_heads):
        if seen is not None:
            seen['num_heads'] = num_heads
        q_len = hidden_states.size(1)
        return (
            torch.zeros(q_len, num_heads, 2),
            torch.zeros(q_len, 1, 2),
            torch.zeros(q_len, 1, 1),
            torch.zeros(q_len, num_heads, 1),
            torch.zeros(q_len, 1, 1),
            torch.zeros(q_len, 1, 1),
        )

    dist_config = SimpleNamespace(dp=dp, attn_tp=attn_tp)
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v32.get_dist_manager',
                        lambda: SimpleNamespace(current_config=lambda: dist_config,
                                                current_context=lambda: SimpleNamespace(dist_config=dist_config)))
    attn.num_heads = 1
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    attn._qkv_proj = fake_qkv_proj
    attn.apply_rotary_pos_emb = lambda q, k, cos, sin, inplace=False: (q, k)
    attn.vc = lambda attn_output, out: out.copy_(attn_output)
    attn.v_head_dim = 1
    attn.o_proj = nn.Identity()


def _make_kv_b_loader_model(fp8_quant_scope=None, input_dim=3):
    model = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(qk_nope_head_dim=2, v_head_dim=2)
    model.quantization_config = {'quant_method': 'fp8'}
    if fp8_quant_scope is not None:
        model.quantization_config['fp8_quant_scope'] = fp8_quant_scope
    model._load_buffers = {}
    prefix = 'model.layers.0.self_attn'
    params = {
        f'{prefix}.kc.weight': nn.Parameter(torch.empty(1, 2, input_dim, dtype=torch.bfloat16)),
        f'{prefix}.vc.weight': nn.Parameter(torch.empty(1, input_dim, 2, dtype=torch.bfloat16)),
    }
    return model, prefix, params


def _split_kv_b_weight(weight):
    weight = weight.unflatten(0, (-1, 4))
    weight_kc, weight_vc = weight.split([2, 2], dim=1)
    return weight_kc, weight_vc.transpose(1, 2).contiguous()


def test_glm_moe_only_loads_kv_b_proj_as_bf16():
    model, prefix, params = _make_kv_b_loader_model(fp8_quant_scope='moe_only')
    loaded_weight = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)

    model._load_weight_attention(
        name=f'{prefix}.kv_b_proj.weight',
        loaded_weight=loaded_weight,
        params_dict=params,
        update_pe_mapping=[],
    )

    expected_kc, expected_vc = _split_kv_b_weight(loaded_weight)
    assert torch.equal(params[f'{prefix}.kc.weight'], expected_kc)
    assert torch.equal(params[f'{prefix}.vc.weight'], expected_vc)
    assert model._load_buffers == {}


def test_global_fp8_loads_kv_b_proj_after_weight_scale():
    model, prefix, params = _make_kv_b_loader_model(input_dim=4)
    loaded_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
    loaded_scale = torch.full((1, 1), 0.5, dtype=torch.float32)

    model._load_weight_attention(
        name=f'{prefix}.kv_b_proj.weight',
        loaded_weight=loaded_weight,
        params_dict=params,
        update_pe_mapping=[],
    )
    assert f'{prefix}.kv_b_proj.weight' in model._load_buffers

    model._load_weight_attention(
        name=f'{prefix}.kv_b_proj.weight_scale_inv',
        loaded_weight=loaded_scale,
        params_dict=params,
        update_pe_mapping=[],
    )

    dequantized_weight = (loaded_weight.float() * loaded_scale).to(torch.bfloat16)
    expected_kc, expected_vc = _split_kv_b_weight(dequantized_weight)
    assert torch.equal(params[f'{prefix}.kc.weight'], expected_kc)
    assert torch.equal(params[f'{prefix}.vc.weight'], expected_vc)
    assert model._load_buffers == {}


@pytest.mark.parametrize('n_heads', [32, 64])
@pytest.mark.parametrize('load_scale_first', [False, True])
def test_fp8_indexer_wk_loads_into_fused_bf16_projection(load_scale_first, n_heads):
    model = DeepseekV32ForCausalLM.__new__(DeepseekV32ForCausalLM)
    nn.Module.__init__(model)
    model._load_buffers = {}
    prefix = 'model.layers.0.self_attn.indexer'
    fused = nn.Parameter(torch.zeros(128 + n_heads, 256, dtype=torch.bfloat16))
    params = {f'{prefix}.wk_weights_proj.weight': fused}

    wk = (torch.arange(128 * 256).reshape(128, 256) % 31 - 15).float().to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    gate = torch.arange(n_heads * 256, dtype=torch.float32).reshape(n_heads, 256).to(torch.bfloat16)
    wk_tensors = [
        (f'{prefix}.wk.weight', wk),
        (f'{prefix}.wk.weight_scale_inv', scale),
    ]
    if load_scale_first:
        wk_tensors.reverse()

    for name, tensor in wk_tensors:
        model._load_weight_attention(name, tensor, params, update_pe_mapping=[])
    model._load_weight_attention(f'{prefix}.weights_proj.weight', gate, params, update_pe_mapping=[])

    expected_wk = _dequantize_blocked_fp8(wk, scale, torch.bfloat16)
    assert torch.equal(fused[:128], expected_wk)
    assert torch.equal(fused[128:], gate)
    assert model._load_buffers == {}


def test_bf16_indexer_wk_loads_directly_into_fused_projection():
    model = DeepseekV32ForCausalLM.__new__(DeepseekV32ForCausalLM)
    nn.Module.__init__(model)
    model._load_buffers = {}
    prefix = 'model.layers.0.self_attn.indexer'
    fused = nn.Parameter(torch.zeros(160, 8, dtype=torch.bfloat16))
    params = {f'{prefix}.wk_weights_proj.weight': fused}
    wk = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8).to(torch.bfloat16)

    model._load_weight_attention(f'{prefix}.wk.weight', wk, params, update_pe_mapping=[])

    assert torch.equal(fused[:128], wk)
    assert model._load_buffers == {}


def test_indexer_rope_interleave_uses_interleaved_layout():
    indexer = Indexer.__new__(Indexer)
    indexer.apply_rotary_pos_emb = DefaultApplyRotaryEmbImpl(interleaved=True).forward

    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    key = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    cos = torch.tensor([[10.0, 20.0, 10.0, 20.0]])
    sin = torch.tensor([[1.0, 2.0, 1.0, 2.0]])

    query_out, key_out = indexer._apply_rotary_pos_emb(query, key, (cos, sin))

    key = key[..., None, :]
    interleaved_cos = torch.tensor([[10.0, 10.0, 20.0, 20.0]]).unsqueeze(-2)
    interleaved_sin = torch.tensor([[1.0, 1.0, 2.0, 2.0]]).unsqueeze(-2)
    expected_query = query * interleaved_cos + rotate_interleaved(query) * interleaved_sin
    expected_key = key * interleaved_cos + rotate_interleaved(key) * interleaved_sin
    assert torch.equal(query_out, expected_query)
    assert torch.equal(key_out, expected_key)


def test_full_indexer_layer_writes_compact_output_and_shared_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 0
    attn.indexer_output_idx = 0
    _patch_minimal_attention(attn, monkeypatch)
    computed_topk = torch.tensor([[4, 2, 1], [3, 2, 0]], dtype=torch.int32)
    attn.indexer = lambda *args, **kwargs: computed_topk
    seen = {}

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(args[0].size(0), 1, 1)

    attn.attn_fwd = fake_attn_fwd
    topk_buffer = DSATopKIndicesBuffer(topk=3)
    all_indexer_topk = torch.full((2, 1, 3), -1, dtype=torch.int32)

    output = attn(
        hidden_states=torch.zeros(1, 2, 2),
        rotary_pos_emb=(torch.zeros(2, 1), torch.zeros(2, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=topk_buffer,
        all_indexer_topk=all_indexer_topk,
    )

    assert output.shape == (1, 2, 1)
    assert torch.equal(topk_buffer.indices[:2], computed_topk)
    assert seen['nsa_indices'].data_ptr() == topk_buffer.indices[:2].data_ptr()
    assert torch.equal(all_indexer_topk[:, 0], computed_topk)


def test_mtp_skips_indexer_and_reads_compacted_topk(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 5
    attn.indexer_output_idx = None
    _patch_minimal_attention(attn, monkeypatch)
    attn.indexer = lambda *args, **kwargs: pytest.fail('recurrent MTP must reuse seed top-k')
    seen = {}

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(args[0].size(0), 1, 1)

    attn.attn_fwd = fake_attn_fwd
    topk_buffer = DSATopKIndicesBuffer(topk=3)
    topk_buffer.write(torch.tensor([[1, 2, 3], [11, 12, 13], [21, 22, 23]], dtype=torch.int32))
    topk_buffer.compact(torch.tensor([2, 0]))

    attn(
        hidden_states=torch.zeros(1, 2, 2),
        rotary_pos_emb=(torch.zeros(2, 1), torch.zeros(2, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=topk_buffer,
        skip_topk=True,
    )

    assert torch.equal(seen['nsa_indices'], torch.tensor([[21, 22, 23], [1, 2, 3]], dtype=torch.int32))


@pytest.mark.parametrize('dp,attn_tp,expected_num_heads', [(1, 2, 2), (2, 2, 4)])
def test_attention_uses_dp_aware_num_heads(monkeypatch, dp, attn_tp, expected_num_heads):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 0
    seen = {}
    _patch_minimal_attention(attn, monkeypatch, dp=dp, attn_tp=attn_tp, seen=seen)
    attn.num_heads = 4
    attn.indexer = lambda *args, **kwargs: torch.tensor([[4, 2, 1]], dtype=torch.int32)
    attn.attn_fwd = lambda *args, **kwargs: torch.zeros(args[0].size(0), args[0].size(1), 1)

    attn(
        hidden_states=torch.zeros(1, 1, 2),
        rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=DSATopKIndicesBuffer(topk=3),
    )

    assert seen['num_heads'] == expected_num_heads


@pytest.mark.parametrize('dp,attn_tp,expected_num_heads', [(1, 2, 2), (2, 2, 4)])
def test_bmm_uses_dp_aware_weight_layout(monkeypatch, dp, attn_tp, expected_num_heads):
    dist_config = SimpleNamespace(dp=dp, attn_tp=attn_tp)
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v2.get_dist_manager',
                        lambda: SimpleNamespace(current_config=lambda: dist_config,
                                                current_context=lambda: SimpleNamespace(dist_config=dist_config)))
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v2.get_tp_world_rank',
                        lambda layer_type=None: (attn_tp, 0))

    bmm = DeepseekV2BMM(batch=4, in_features=2, out_features=3, dtype=torch.float32, device='cpu')
    weight = torch.arange(24, dtype=torch.float32).view(4, 2, 3)
    bmm.weight.weight_loader(bmm.weight, weight)

    expected = weight if dp > 1 else weight.chunk(attn_tp, 0)[0]
    assert bmm.weight.shape == (expected_num_heads, 2, 3)
    torch.testing.assert_close(bmm.weight, expected)


def test_shared_indexer_layer_reuses_shared_topk_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 3
    attn.indexer = None
    attn.indexer_output_idx = None
    _patch_minimal_attention(attn, monkeypatch)
    seen = {}

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(args[0].size(0), 1, 1)

    attn.attn_fwd = fake_attn_fwd
    topk_buffer = DSATopKIndicesBuffer(topk=3)
    shared_topk = torch.tensor([[5, 4, 3]], dtype=torch.int32)
    topk_buffer.write(shared_topk)
    all_indexer_topk = torch.full((1, 1, 3), -1, dtype=torch.int32)

    output = attn(
        hidden_states=torch.zeros(1, 1, 2),
        rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=topk_buffer,
        all_indexer_topk=all_indexer_topk,
    )

    assert output.shape == (1, 1, 1)
    assert seen['nsa_indices'].data_ptr() == topk_buffer.indices[:1].data_ptr()
    assert torch.all(all_indexer_topk == -1)


def test_attention_requires_shared_topk_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 3
    attn.indexer = None
    attn.num_heads = 1
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    dist_config = SimpleNamespace(dp=1, attn_tp=1)
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v32.get_dist_manager',
                        lambda: SimpleNamespace(current_config=lambda: dist_config,
                                                current_context=lambda: SimpleNamespace(dist_config=dist_config)))
    attn._qkv_proj = lambda *args, **kwargs: (
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
    )
    attn.apply_rotary_pos_emb = lambda q, k, cos, sin, inplace=False: (q, k)

    with pytest.raises(RuntimeError, match='requires a DSA top-k indices buffer'):
        attn(
            hidden_states=torch.zeros(1, 1, 2),
            rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
            past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        )


def test_layer_indexer_type_defaults_to_full_and_reads_shared_entries():
    config = SimpleNamespace(num_hidden_layers=3, indexer_types=['full', 'shared'])

    assert get_layer_indexer_type(config, 0) == 'full'
    assert get_layer_indexer_type(config, 1) == 'shared'
    assert get_layer_indexer_type(config, 2) == 'full'
    assert get_layer_indexer_type(SimpleNamespace(indexer_types=None), 1) == 'full'
    assert get_full_indexer_layer_ids(config) == (0, 2)
    assert get_full_indexer_layer_ids(SimpleNamespace(num_hidden_layers=3, indexer_types=None)) == (0, 1, 2)


def test_moe_gate_captures_logical_experts_before_eplb_mapping(monkeypatch):
    class FakeTopK(nn.Module):

        def forward(self, logits):
            logical_ids = torch.tensor([[3, 1], [2, 0]], device=logits.device)
            return torch.ones_like(logical_ids, dtype=torch.float32), logical_ids

    gate = MoEGate.__new__(MoEGate)
    nn.Module.__init__(gate)
    gate.weight = nn.Parameter(torch.zeros(4, 2))
    gate.fake_eplb = False
    gate.topk_method = 'greedy'
    gate.renormalize = False
    gate.routed_scaling_factor = 1.0
    gate.softmax_topk = FakeTopK()
    gate.eplb_dispatch_info = object()
    monkeypatch.setattr(
        'lmdeploy.pytorch.models.deepseek_v2.EPLBManager.topk_ids_logical_to_physical',
        lambda ids, info: ids + 10,
    )
    captured = torch.full((2, 2), torch.iinfo(torch.uint16).max, dtype=torch.uint16)

    _, dispatch_ids = gate(torch.zeros(2, 2), routed_experts=captured)

    expected = torch.tensor([[3, 1], [2, 0]])
    assert torch.equal(captured, expected.to(torch.uint16))
    assert torch.equal(dispatch_ids, expected + 10)


def test_deepseek_moe_writes_only_its_routed_expert_layer():
    class FakeGate(nn.Module):

        def forward(self, hidden_states, routed_experts=None):
            ids = torch.tensor([[3, 1], [2, 0]], device=hidden_states.device)
            if routed_experts is not None:
                routed_experts.copy_(ids)
            return torch.ones_like(ids, dtype=torch.float32), ids

    class FakeExperts(nn.Module):

        def forward(self, hidden_states, topk_weights, topk_ids):
            return hidden_states

    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.layer_idx = 3
    moe.hidden_dim = 2
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = None
    moe._all_reduce = False
    sentinel = torch.iinfo(torch.uint16).max
    all_routed_experts = torch.full((2, 5, 2), sentinel, dtype=torch.uint16)

    output = moe(torch.zeros(1, 2, 2), all_routed_experts=all_routed_experts)

    assert output.shape == (1, 2, 2)
    assert torch.equal(all_routed_experts[:, 3], torch.tensor([[3, 1], [2, 0]], dtype=torch.uint16))
    assert torch.all(all_routed_experts[:, :3] == sentinel)
    assert torch.all(all_routed_experts[:, 4] == sentinel)


def test_glm52_causal_lm_returns_routed_experts_and_indexer_topk(monkeypatch):
    class FakeModel(nn.Module):

        def forward(self, input_ids, all_routed_experts=None, all_indexer_topk=None, **kwargs):
            if all_routed_experts is not None:
                all_routed_experts[:, 3].fill_(7)
                all_routed_experts[:, 4].fill_(9)
            if all_indexer_topk is not None:
                all_indexer_topk.fill_(5)
            return torch.zeros(1, input_ids.size(1), 4)

        forward_microbatch = forward

    context = SimpleNamespace(enable_microbatch=False)
    monkeypatch.setattr(
        'lmdeploy.pytorch.models.deepseek_v32.get_step_ctx_manager',
        lambda: SimpleNamespace(current_context=lambda: context),
    )
    model = DeepseekV32ForCausalLM.__new__(DeepseekV32ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=5,
                                   num_experts_per_tok=8,
                                   index_topk=3,
                                   indexer_types=['full', 'full', 'full', 'shared', 'shared'])
    model.enable_return_routed_experts = True
    model.enable_return_indexer_topk = True
    model.num_indexer_layers = 3
    model.model = FakeModel()

    outputs = model(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        position_ids=torch.arange(2)[None],
        past_key_values=[],
    )

    routed_experts = outputs['all_routed_experts']
    assert routed_experts.dtype == torch.uint16
    assert routed_experts.shape == (2, 5, 8)
    assert torch.all(routed_experts[:, :3] == torch.iinfo(torch.uint16).max)
    assert torch.all(routed_experts[:, 3] == 7)
    assert torch.all(routed_experts[:, 4] == 9)
    assert outputs['all_indexer_topk'].shape == (2, 3, 3)
    assert torch.all(outputs['all_indexer_topk'] == 5)
