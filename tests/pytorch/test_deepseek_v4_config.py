# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.hf_configs import config_from_pretrained
from lmdeploy.pytorch.config import DistConfig, ModelConfig
from lmdeploy.pytorch.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    ParallelEmbedding,
    QuantLinear,
    V4Args,
    _dequantize_wo_a_shard,
    _load_vector_shard,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from lmdeploy.pytorch.models.module_map import MODULE_MAP

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


def test_deepseek_v4_has_cudagraph_interface():
    assert hasattr(DeepseekV4ForCausalLM, 'make_output_buffers')
    assert hasattr(DeepseekV4ForCausalLM, 'make_buffers_cudagraph')
    assert hasattr(DeepseekV4ForCausalLM, 'fill_buffers_cudagraph')


def test_deepseek_v4_cuda_graph_disabled():
    assert DeepseekV4ForCausalLM.support_cuda_graph is not None
    model = object.__new__(DeepseekV4ForCausalLM)
    assert model.support_cuda_graph(None, None, None, attn_metadata=None) is False


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
