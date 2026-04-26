# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.hf_configs import config_from_pretrained
from lmdeploy.pytorch.backends.cuda.moe.v4_fp4 import _slice_local_topk, _v4_swiglu
from lmdeploy.pytorch.config import DistConfig, ModelConfig
from lmdeploy.pytorch.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    MoE,
    ParallelEmbedding,
    QuantLinear,
    V4Args,
    _build_prefix_positions,
    _build_topk_range,
    _build_window_positions,
    _dequantize_wo_a_shard,
    _load_vector_shard,
    _map_v4_expert_param_name,
    _next_power_of_2,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from lmdeploy.pytorch.models.module_map import MODULE_MAP
from lmdeploy.pytorch.nn.moe.v4_fp4 import (
    FusedMoEV4,
    V4ExpertTPWeights,
    _convert_fp4_to_blocked_fp8,
    _dequantize_fp4_weight,
    _get_v4_moe_runtime_kind,
)
from lmdeploy.pytorch.nn.quant_utils import quant_blocked_fp8

MODEL_PATH = '/nvme1/yaoqian/space/tmp/lmdeploy_test/develop/dsv4/DeepSeek-V4-Flash'


def test_deepseek_v4_config_from_pretrained():
    cfg = config_from_pretrained(MODEL_PATH, trust_remote_code=True)
    assert cfg.model_type == 'deepseek_v4'
    assert cfg.architectures[0] == 'DeepseekV4ForCausalLM'
    assert cfg.compress_ratios


def test_deepseek_v4_model_config_builder():
    cfg = ModelConfig.from_pretrained(MODEL_PATH,
                                      trust_remote_code=True,
                                      device_type='cpu',
                                      dist_config=DistConfig(tp=2))
    assert cfg.hf_config.model_type == 'deepseek_v4'
    assert cfg.states_shapes
    assert cfg.num_attention_heads == 2
    assert cfg.num_key_value_heads == 2
    assert cfg.check_env_func is not None


def test_deepseek_v4_module_map():
    assert MODULE_MAP['DeepseekV4ForCausalLM'] == 'lmdeploy.pytorch.models.deepseek_v4.DeepseekV4ForCausalLM'


def test_deepseek_v4_get_model_arch():
    arch, cfg = get_model_arch(MODEL_PATH)
    assert arch == 'DeepseekV4ForCausalLM'
    assert cfg.model_type == 'deepseek_v4'
    assert hasattr(cfg, 'max_position_embeddings')


def test_deepseek_v4_args_n_groups_alias():
    cfg = config_from_pretrained(MODEL_PATH, trust_remote_code=True)
    args = V4Args(
        dim=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=cfg.num_hidden_layers,
        n_hash_layers=cfg.num_hash_layers,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=tuple(cfg.compress_ratios),
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
        rope_factor=cfg.rope_scaling['factor'],
        beta_fast=cfg.rope_scaling['beta_fast'],
        beta_slow=cfg.rope_scaling['beta_slow'],
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
    )
    assert args.n_groups == cfg.o_groups


def test_deepseek_v4_quant_linear_casts_fp8_input_to_bf16():

    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            assert x.dtype == torch.bfloat16
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

    layer = QuantLinear(16, 8, _Kernel(), dtype=torch.float8_e4m3fn, device='cpu')
    out = layer(torch.randn(2, 16, dtype=torch.float32))
    assert out.shape == (2, 8)


def test_deepseek_v4_moe_runtime_kind_prefers_fp8_on_cpu():
    assert _get_v4_moe_runtime_kind(torch.device('cpu')) == 'fp8'


def test_deepseek_v4_moe_runtime_kind_dispatches_by_arch(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (9, 0))
    assert _get_v4_moe_runtime_kind(torch.device('cuda')) == 'fp8'
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (10, 0))
    assert _get_v4_moe_runtime_kind(torch.device('cuda')) == 'fp4'


def test_deepseek_v4_fp4_to_blocked_fp8_conversion_matches_blocked_quantization():
    weight = torch.randint(0, 255, (128, 64), dtype=torch.uint8).view(torch.int8)
    raw_scale = torch.pow(2, torch.randint(0, 8, (128, 4), dtype=torch.int32)).to(torch.float32)
    scale = raw_scale.to(torch.float8_e8m0fnu)

    dense_weight = _dequantize_fp4_weight(weight, scale)
    ref_weight, ref_scale = quant_blocked_fp8(dense_weight, torch.float8_e4m3fn, 128, scale_fmt='ue8m0')

    batched_weight, batched_scale = _convert_fp4_to_blocked_fp8(weight.unsqueeze(0), scale.unsqueeze(0))
    assert torch.equal(batched_weight[0], ref_weight)
    assert torch.equal(batched_scale[0], ref_scale)


def test_deepseek_v4_expert_param_mapping_for_fused_and_legacy():
    fused_gate = _map_v4_expert_param_name('model.layers.0.ffn.experts.3.w1.weight', True)
    fused_up = _map_v4_expert_param_name('model.layers.0.ffn.experts.3.w3.scale', True)
    fused_down = _map_v4_expert_param_name('model.layers.0.ffn.experts.3.w2.weight', True)
    legacy_gate = _map_v4_expert_param_name('model.layers.0.ffn.experts.3.w1.weight', False)

    assert fused_gate == ('model.layers.0.ffn.experts.ckpt_gate_up.weight', 'gate')
    assert fused_up == ('model.layers.0.ffn.experts.ckpt_gate_up.scale', 'up')
    assert fused_down == ('model.layers.0.ffn.experts.ckpt_down.weight', 'down')
    assert legacy_gate == ('model.layers.0.ffn.experts.3.w1.weight', 'gate')


def test_deepseek_v4_tp_weights_shard_intermediate_dim(monkeypatch):
    monkeypatch.setattr('lmdeploy.pytorch.nn.moe.v4_fp4.get_tp_world_rank', lambda name='moe': (3, 2))
    mod = V4ExpertTPWeights(num_experts=4, hidden_dim=256, ffn_dim=640, weight_type='gate_up', device='cpu')

    loaded_gate = torch.arange(640 * 128, dtype=torch.int8).view(640, 128)
    mod.weight_loader(mod.weight, loaded_gate, expert_id=1, shard_id='gate')
    ref_gate = loaded_gate.split([256, 256, 128], dim=0)[2]
    assert torch.equal(mod.weight.data[1, :128], ref_gate)

    mod_down = V4ExpertTPWeights(num_experts=4, hidden_dim=256, ffn_dim=640, weight_type='down', device='cpu')
    loaded_down = torch.arange(256 * 320, dtype=torch.int8).view(256, 320)
    mod_down.weight_loader(mod_down.weight, loaded_down, expert_id=1, shard_id='down')
    ref_down = loaded_down.split([128, 128, 64], dim=1)[2]
    assert torch.equal(mod_down.weight.data[1], ref_down)


def test_deepseek_v4_fused_moe_owns_nested_update_weights(monkeypatch):

    class _FakeImpl(torch.nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.block_size = 128
            self.calls = 0
            self.gate_up = type('GateUp', (), {'update_weight': lambda self, w, s: None})()
            self.down = type('Down', (), {'update_weight': lambda self, w, s: None})()

        def update_weights(self):
            self.calls += 1

        def forward(self, hidden_states, topk_weights, topk_ids):
            return hidden_states

    monkeypatch.setattr('lmdeploy.pytorch.nn.moe.v4_fp4._get_v4_moe_runtime_kind', lambda device: 'fp8')
    monkeypatch.setattr('lmdeploy.pytorch.nn.moe.v4_fp4.FusedMoEBlockedF8', _FakeImpl)
    monkeypatch.setattr('lmdeploy.pytorch.nn.moe.v4_fp4._convert_fp4_to_blocked_fp8',
                        lambda weight, scale, block_size=128: (
                            torch.zeros((weight.size(0), weight.size(1), weight.size(2) * 2),
                                        dtype=torch.float8_e4m3fn),
                            torch.zeros((weight.size(0), 1, (weight.size(2) * 2) // block_size),
                                        dtype=torch.float32),
                        ))

    moe = FusedMoEV4(hidden_dim=256, ffn_dim=256, num_experts=4, top_k=2, device=torch.device('cpu'))
    nested_update = moe.impl.update_weights
    moe.update_weights()
    assert moe._impl_update_weights is not None
    assert moe.impl.calls == 1
    nested_update()
    assert moe.impl.calls == 1


def test_deepseek_v4_legacy_expert_loader_ignores_expert_kwargs(monkeypatch):
    calls = []

    def _fake_load_weight(param, loaded_weight, **kwargs):
        calls.append(kwargs)

    model = object.__new__(DeepseekV4ForCausalLM)
    model.world_size = 1
    model.rank = 0
    model.layers = [type('Layer', (), {'ffn': type('FFN', (), {'use_fused_experts': False})()})()]

    legacy_param = torch.nn.Parameter(torch.empty(2, 2))
    model.named_parameters = lambda: [('layers.0.ffn.experts.3.w1.weight', legacy_param)]

    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v4.load_weight', _fake_load_weight)
    model.load_weights([('layers.0.ffn.experts.3.w1.weight', torch.ones(2, 2))])

    assert calls == [{}]


def test_deepseek_v4_moe_swiglu_matches_v4_clamp_rules():
    x = torch.tensor([[12.0, -20.0, 30.0, -30.0]], dtype=torch.bfloat16)
    out = _v4_swiglu(x, swiglu_limit=10.0)
    gate = torch.tensor([[10.0, -20.0]], dtype=torch.float32)
    up = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    ref = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    assert out.shape == (1, 2)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out, ref)


def test_deepseek_v4_local_topk_sanitizes_invalid_ids_for_blocked_fp8():
    topk_ids = torch.tensor([[3, 70, 5], [90, 6, 2]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]], dtype=torch.float32)
    local_ids, local_weights, local_mask = _slice_local_topk(topk_ids,
                                                             topk_weights,
                                                             expert_offset=0,
                                                             num_experts=8,
                                                             invalid_expert=0)
    assert local_ids.min().item() >= 0
    assert torch.equal(local_mask,
                       torch.tensor([[True, False, True], [False, True, True]]))
    assert torch.equal(local_weights,
                       torch.tensor([[0.3, 0.0, 0.3], [0.0, 0.5, 0.3]], dtype=torch.float32))


def test_deepseek_v4_local_topk_keeps_negative_sentinel_for_fp4_grouped():
    topk_ids = torch.tensor([[32, 35], [40, 31]], dtype=torch.int64)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)
    local_ids, local_weights, local_mask = _slice_local_topk(topk_ids,
                                                             topk_weights,
                                                             expert_offset=32,
                                                             num_experts=4,
                                                             invalid_expert=-1)
    assert torch.equal(local_ids, torch.tensor([[0, 3], [-1, -1]], dtype=torch.int64))
    assert torch.equal(local_weights, torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32))
    assert torch.equal(local_mask, torch.tensor([[True, True], [False, False]]))


def test_deepseek_v4_quant_linear_uses_bf16_default_dtype_for_fp8_gemm():

    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            assert torch.get_default_dtype() == torch.bfloat16
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

    prev = torch.get_default_dtype()
    layer = QuantLinear(16, 8, _Kernel(), dtype=torch.float8_e4m3fn, device='cpu')
    out = layer(torch.randn(2, 16, dtype=torch.float32))
    assert out.shape == (2, 8)
    assert torch.get_default_dtype() == prev


def test_deepseek_v4_quant_linear_uses_bf16_default_dtype_for_fp4_gemm():

    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            assert torch.get_default_dtype() == torch.bfloat16
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

    prev = torch.get_default_dtype()
    layer = QuantLinear(16, 8, _Kernel(), dtype=torch.float4_e2m1fn_x2, device='cpu')
    out = layer(torch.randn(2, 16, dtype=torch.float32))
    assert out.shape == (2, 8)
    assert torch.get_default_dtype() == prev


def test_deepseek_v4_parallel_embedding_uses_model_dtype():
    emb = ParallelEmbedding(16, 4, world_size=1, rank=0, device='cpu', dtype=torch.bfloat16)
    assert emb.weight.dtype == torch.bfloat16


def test_deepseek_v4_shared_expert_uses_checkpoint_quant_dtype():
    from lmdeploy.pytorch.models.deepseek_v4 import Expert

    class _Kernel:
        pass

    expert = Expert(16, 8, _Kernel(), dtype=torch.float8_e4m3fn, device='cpu')
    assert expert.w1.weight.dtype == torch.float8_e4m3fn
    assert expert.w2.weight.dtype == torch.float8_e4m3fn
    assert expert.w3.weight.dtype == torch.float8_e4m3fn


def test_deepseek_v4_moe_defaults_to_fused_experts(monkeypatch):
    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

    monkeypatch.delenv('LMDEPLOY_DSV4_EXPERIMENTAL_FUSED_MOE', raising=False)
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v4.FusedMoEV4',
                        lambda *args, **kwargs: torch.nn.Identity())
    args = V4Args(dim=16,
                  n_heads=2,
                  vocab_size=32,
                  moe_inter_dim=8,
                  n_layers=1,
                  n_hash_layers=0,
                  n_routed_experts=4,
                  n_shared_experts=1,
                  n_activated_experts=2,
                  score_func='sigmoid',
                  route_scale=1.0,
                  swiglu_limit=0.0,
                  q_lora_rank=8,
                  head_dim=8,
                  rope_head_dim=4,
                  norm_eps=1e-5,
                  o_groups=2,
                  o_lora_rank=4,
                  window_size=16,
                  compress_ratios=(0,),
                  compress_rope_theta=10000.0,
                  original_seq_len=1024,
                  rope_theta=10000.0,
                  rope_factor=1.0,
                  beta_fast=32,
                  beta_slow=1,
                  index_n_heads=2,
                  index_head_dim=8,
                  index_topk=2,
                  hc_mult=1,
                  hc_sinkhorn_iters=1,
                  hc_eps=1e-6)
    moe = MoE(layer_id=0, args=args, kernel_mod=_Kernel(), world_size=1, rank=0, device='cpu')
    assert moe.use_fused_experts is True
    assert isinstance(moe.experts, torch.nn.Identity)


def test_deepseek_v4_moe_env_can_force_legacy_experts(monkeypatch):
    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setenv('LMDEPLOY_DSV4_EXPERIMENTAL_FUSED_MOE', '0')
    args = V4Args(dim=16,
                  n_heads=2,
                  vocab_size=32,
                  moe_inter_dim=8,
                  n_layers=1,
                  n_hash_layers=0,
                  n_routed_experts=4,
                  n_shared_experts=1,
                  n_activated_experts=2,
                  score_func='sigmoid',
                  route_scale=1.0,
                  swiglu_limit=0.0,
                  q_lora_rank=8,
                  head_dim=8,
                  rope_head_dim=4,
                  norm_eps=1e-5,
                  o_groups=2,
                  o_lora_rank=4,
                  window_size=16,
                  compress_ratios=(0,),
                  compress_rope_theta=10000.0,
                  original_seq_len=1024,
                  rope_theta=10000.0,
                  rope_factor=1.0,
                  beta_fast=32,
                  beta_slow=1,
                  index_n_heads=2,
                  index_head_dim=8,
                  index_topk=2,
                  hc_mult=1,
                  hc_sinkhorn_iters=1,
                  hc_eps=1e-6)
    moe = MoE(layer_id=0, args=args, kernel_mod=_Kernel(), world_size=1, rank=0, device='cpu')
    assert moe.use_fused_experts is False
    assert isinstance(moe.experts, torch.nn.ModuleDict)


def test_deepseek_v4_has_cudagraph_interface():
    assert hasattr(DeepseekV4ForCausalLM, 'make_output_buffers')
    assert hasattr(DeepseekV4ForCausalLM, 'make_buffers_cudagraph')
    assert hasattr(DeepseekV4ForCausalLM, 'fill_buffers_cudagraph')


def test_deepseek_v4_cuda_graph_support_is_decode_q1_only():
    assert DeepseekV4ForCausalLM.support_cuda_graph is not None
    model = object.__new__(DeepseekV4ForCausalLM)
    model.layers = [type('Layer', (), {'ffn': type('FFN', (), {'use_fused_experts': True})()})()]
    model.ctx_mgr = type('CtxMgr', (), {'current_context': lambda self: None})()
    attn_metadata = type('Meta', (), {'is_decoding': True, 'q_seqlens': torch.ones(4, dtype=torch.int32)})()
    assert model.support_cuda_graph(torch.zeros((1, 4), dtype=torch.long), None, None, attn_metadata=attn_metadata)
    attn_metadata.is_decoding = False
    assert model.support_cuda_graph(torch.zeros((1, 4), dtype=torch.long), None, None, attn_metadata=attn_metadata) is False    # noqa: E501


def test_deepseek_v4_graph_key_extends_with_history_bucket():
    model = object.__new__(DeepseekV4ForCausalLM)
    graph_key = model.get_graph_key_cudagraph((8, True, False, 1), history_lengths=torch.tensor([3, 9, 4]))
    assert graph_key == (8, True, False, 1, 16)


def test_deepseek_v4_get_logits_preserves_batch_dim():
    model = object.__new__(DeepseekV4ForCausalLM)
    model._hc_head = lambda x: x
    model.norm = lambda x: x

    class _Head:

        def get_logits(self, x):
            return x

    model.head = _Head()

    hidden_states = torch.randn(1, 2, 4)
    logits = model.get_logits(hidden_states)
    assert logits.shape == (1, 2, 4)


def test_deepseek_v4_topk_indices_are_3d():
    window = get_window_topk_idxs(window_size=4, bsz=1, seqlen=1, start_pos=5, device='cpu')
    compressed = get_compress_topk_idxs(ratio=4, bsz=1, seqlen=1, start_pos=5, offset=8, device='cpu')
    assert window.ndim == 3
    assert compressed.ndim == 3


def test_deepseek_v4_decode_position_helpers():
    positions, mask = _build_prefix_positions(torch.tensor([0, 2, 4]), 4)
    assert torch.equal(positions, torch.tensor([[-1, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, 3]]))
    assert torch.equal(mask, positions >= 0)

    window_positions, window_lens, window_mask = _build_window_positions(torch.tensor([1, 3, 6]), 4)
    assert torch.equal(window_lens, torch.tensor([1, 3, 4]))
    assert torch.equal(window_positions, torch.tensor([[0, -1, -1, -1], [0, 1, 2, -1], [2, 3, 4, 5]]))
    assert torch.equal(window_mask, window_positions >= 0)

    topk = _build_topk_range(torch.tensor([1, 3]), 4, offset=8)
    assert torch.equal(topk, torch.tensor([[[8, -1, -1, -1]], [[8, 9, 10, -1]]]))


def test_deepseek_v4_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(9) == 16


def test_deepseek_v4_compressor_rotate_path_quantizes():
    called = {'rotate': False, 'quant': False}

    class _Kernel:

        @staticmethod
        def act_quant(x, *args, **kwargs):
            return x, torch.ones((1, 1), dtype=torch.float8_e8m0fnu)

        @staticmethod
        def fp8_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_gemm(qx, scale, weight, weight_scale, scale_dtype):
            return torch.zeros((*qx.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)

        @staticmethod
        def fp4_act_quant(x, *args, **kwargs):
            called['quant'] = True
            x.add_(1)

    cfg = config_from_pretrained(MODEL_PATH, trust_remote_code=True)
    args = V4Args(
        dim=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=cfg.num_hidden_layers,
        n_hash_layers=cfg.num_hash_layers,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=tuple(cfg.compress_ratios),
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
        rope_factor=cfg.rope_scaling['factor'],
        beta_fast=cfg.rope_scaling['beta_fast'],
        beta_slow=cfg.rope_scaling['beta_slow'],
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
    )
    import lmdeploy.pytorch.models.deepseek_v4 as deepseek_v4_mod
    from lmdeploy.pytorch.models.deepseek_v4 import Compressor
    comp = Compressor(args, _Kernel(), compress_ratio=4, head_dim=8, device='cpu', rotate=True)
    comp.freqs_cis = torch.ones(8, 4, dtype=torch.complex64)
    x = torch.ones(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
    orig_rotate_activation = deepseek_v4_mod.rotate_activation
    try:
        def _rotate(tensor):
            called['rotate'] = True
            return tensor
        deepseek_v4_mod.rotate_activation = _rotate
        comp(x, 0, 0)
    finally:
        deepseek_v4_mod.rotate_activation = orig_rotate_activation
    assert called['rotate'] is True
    assert called['quant'] is True


def test_deepseek_v4_vector_shard_loader():
    param = torch.nn.Parameter(torch.empty(16, dtype=torch.float32), requires_grad=False)
    loaded_weight = torch.arange(64, dtype=torch.float32)
    _load_vector_shard(param, loaded_weight, world_size=4, rank=2)
    assert torch.equal(param, loaded_weight.chunk(4)[2])


def test_deepseek_v4_dequantize_wo_a_shard_matches_official_convert():
    weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
    scale = torch.full((2, 2), 2.0, dtype=torch.float8_e8m0fnu)
    shard = _dequantize_wo_a_shard(weight, scale, world_size=1, rank=0)
    assert shard.shape == (256, 256)
    assert shard.dtype == torch.bfloat16
    assert torch.all(shard == torch.tensor(2.0, dtype=torch.bfloat16))


def test_deepseek_v4_builder_declares_named_cache_specs():
    from lmdeploy.hf_configs.configuration_deepseek_v4 import DeepseekV4Config
    from lmdeploy.pytorch.configurations.deepseek_v4 import DeepseekV4ModelConfigBuilder

    cfg = DeepseekV4Config()
    model_config = DeepseekV4ModelConfigBuilder.build(cfg)

    # Must restore real layer count / head_dim and disable standard kv cache
    assert model_config.num_layers == cfg.num_hidden_layers
    assert model_config.head_dim == cfg.head_dim
    assert model_config.use_standard_kv_cache is False

    # block_cache_specs must be declared
    assert len(model_config.block_cache_specs) >= 1
    spec_names = {s.name for s in model_config.block_cache_specs}
    assert 'v4_raw_kv' in spec_names

    # backward-compat bridge: states_shapes must be derived from state_cache_specs
    assert len(model_config.states_shapes) == len(model_config.state_cache_specs)


def test_deepseek_v4_cache_engine_allocates_named_block_caches():
    from lmdeploy.hf_configs.configuration_deepseek_v4 import DeepseekV4Config
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.configurations.deepseek_v4 import DeepseekV4ModelConfigBuilder
    from lmdeploy.pytorch.engine.cache_engine import CacheEngine

    cfg = DeepseekV4Config()
    model_config = DeepseekV4ModelConfigBuilder.build(cfg)
    cache_config = CacheConfig(
        max_batches=2,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=10,
        num_state_caches=0,
    )
    engine = CacheEngine(cache_config, model_config, world_size=1)

    assert 'v4_raw_kv' in engine.block_caches
    raw_kv = engine.block_caches['v4_raw_kv']
    assert raw_kv.shape == (model_config.num_layers, 10, 64, model_config.head_dim)
    assert raw_kv.dtype == torch.bfloat16
