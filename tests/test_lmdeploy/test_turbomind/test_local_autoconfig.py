from types import SimpleNamespace

import _turbomind as _tm
from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    model_type = 'dummy'


def test_load_model_config_returns_text_config_object(monkeypatch):
    from lmdeploy.turbomind.models import utils

    text_cfg = DummyConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
    )
    outer_cfg = DummyConfig(text_config=text_cfg)

    monkeypatch.setattr(utils, 'get_model_arch', lambda model_path: ('DummyForCausalLM', outer_cfg))

    assert utils.load_model_config('/fake/model') is text_cfg


def test_load_model_config_returns_outer_object_without_text_config(monkeypatch):
    from lmdeploy.turbomind.models import utils

    cfg = DummyConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
    )

    monkeypatch.setattr(utils, 'get_model_arch', lambda model_path: ('DummyForCausalLM', cfg))

    assert utils.load_model_config('/fake/model') is cfg


def test_apply_hf_overrides_updates_config_object():
    from lmdeploy.turbomind.converter import _apply_hf_overrides

    cfg = DummyConfig(hidden_size=16, rope_scaling={'type': 'linear', 'factor': 2.0})

    _apply_hf_overrides(cfg, {
        'hidden_size': 32,
        'rope_scaling': {'factor': 4.0},
        'new_field': 'kept',
    })

    assert cfg.hidden_size == 32
    assert cfg.rope_scaling == {'type': 'linear', 'factor': 4.0}
    assert cfg.new_field == 'kept'


def test_apply_hf_overrides_updates_nested_config_object():
    from lmdeploy.turbomind.converter import _apply_hf_overrides

    cfg = DummyConfig(llm_config=DummyConfig(hidden_size=16, rope_scaling={'type': 'linear', 'factor': 2.0}))

    _apply_hf_overrides(cfg, {
        'llm_config': {
            'hidden_size': 32,
            'rope_scaling': {'factor': 4.0},
        },
    })

    assert cfg.llm_config.hidden_size == 32
    assert cfg.llm_config.rope_scaling == {'type': 'linear', 'factor': 4.0}


def test_parse_rope_param_reads_config_object_fields():
    from lmdeploy.turbomind.models.utils import parse_rope_param

    cfg = DummyConfig(
        rope_theta=500000.0,
        max_position_embeddings=4096,
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'low_freq_factor': 1.0,
            'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        },
    )

    rope, max_pos = parse_rope_param(cfg, head_dim=128)

    assert max_pos == 4096
    assert rope.type == 'llama3'
    assert rope.base == 500000.0
    assert rope.dim == 128
    assert rope.factor == 8.0
    assert rope.low_freq_factor == 1.0
    assert rope.high_freq_factor == 4.0
    assert rope.original_max_position_embeddings == 8192


def test_make_attention_config_reads_only_common_attention_fields():
    from lmdeploy.turbomind.models.utils import make_attention_config

    cfg = DummyConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        max_position_embeddings=128,
    )
    engine_cfg = SimpleNamespace(attn_tp_size=2)

    attn_cfg = make_attention_config(
        cfg,
        engine_cfg,
        data_type=_tm.DataType.TYPE_FP16,
    )

    assert attn_cfg.hidden_dim == 16
    assert attn_cfg.head_num == 4
    assert attn_cfg.kv_head_num == 2
    assert attn_cfg.head_dim == 8
    assert attn_cfg.tp_size == 2
    assert attn_cfg.data_type == _tm.DataType.TYPE_FP16
    assert attn_cfg.rope.dim == 8


def test_make_attention_config_applies_rope_scaling_factor_override():
    from lmdeploy.turbomind.models.utils import make_attention_config

    cfg = DummyConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        rope_theta=10000.0,
        max_position_embeddings=128,
    )
    engine_cfg = SimpleNamespace(attn_tp_size=2, rope_scaling_factor=2.0)

    attn_cfg = make_attention_config(
        cfg,
        engine_cfg,
        data_type=_tm.DataType.TYPE_FP16,
    )

    assert attn_cfg.rope.factor == 2.0
    assert attn_cfg.rope.max_position_embeddings == 128


def test_model_weight_and_ffn_helpers_read_module_fields():
    from lmdeploy.turbomind.builders import _act_type_id
    from lmdeploy.turbomind.models.utils import make_ffn_config, make_model_weight_config

    cfg = DummyConfig(hidden_size=16, intermediate_size=64)
    engine_cfg = SimpleNamespace(attn_tp_size=2, attn_cp_size=1, mlp_tp_size=4)

    root_cfg = make_model_weight_config(
        cfg,
        engine_cfg,
        data_type=_tm.DataType.TYPE_FP16,
    )
    ffn_cfg = make_ffn_config(
        cfg,
        engine_cfg,
        data_type=_tm.DataType.TYPE_FP16,
        act_type=_act_type_id('silu'),
    )

    assert root_cfg.hidden_units == 16
    assert root_cfg.tp_size == 2
    assert root_cfg.data_type == _tm.DataType.TYPE_FP16
    assert ffn_cfg.hidden_dim == 16
    assert ffn_cfg.inter_size == 64
    assert ffn_cfg.tp_size == 4
    assert ffn_cfg.data_type == _tm.DataType.TYPE_FP16
    assert ffn_cfg.act_type == _act_type_id('silu')


def test_make_moe_config_returns_populated_config():
    from lmdeploy.turbomind.models.utils import make_moe_config

    cfg = DummyConfig(hidden_size=16)
    engine_cfg = SimpleNamespace(mlp_tp_size=4)

    moe_cfg = make_moe_config(
        cfg, engine_cfg,
        data_type=_tm.DataType.TYPE_FP16,
        experts_per_token=4,
    )

    assert moe_cfg.method == 1
    assert moe_cfg.experts_per_token == 4
    assert moe_cfg.norm_topk_prob is True
    assert moe_cfg.shared_gate is False
    assert moe_cfg.routed_scale == 1.0
    assert moe_cfg.router_bias is False
    assert moe_cfg.topk_group == 1
    assert moe_cfg.topk_method == 'greedy'
    assert moe_cfg.n_group == 1
    assert moe_cfg.scoring_func == 'softmax'
    assert moe_cfg.router_n_groups == 0
    assert moe_cfg.hidden_dim == 16
    assert moe_cfg.mlp_bias is False
    assert moe_cfg.data_type == _tm.DataType.TYPE_FP16
    assert moe_cfg.tp_size == 4
    assert moe_cfg.act_type == 0  # silu
    assert moe_cfg.fuse_silu is True


def test_make_moe_config_overrides_defaults():
    from lmdeploy.turbomind.models.utils import make_moe_config

    cfg = DummyConfig(hidden_size=32)
    engine_cfg = SimpleNamespace(mlp_tp_size=2)

    moe_cfg = make_moe_config(
        cfg, engine_cfg,
        data_type=_tm.DataType.TYPE_BF16,
        experts_per_token=8,
        act_type=1,
        norm_topk_prob=False,
        shared_gate=True,
        router_bias=True,
        mlp_bias=True,
        topk_method='noaux_tc',
        scoring_func='sigmoid',
        routed_scale=2.0,
        topk_group=2,
        n_group=2,
        router_n_groups=4,
    )

    assert moe_cfg.experts_per_token == 8
    assert moe_cfg.act_type == 1
    assert moe_cfg.norm_topk_prob is False
    assert moe_cfg.shared_gate is True
    assert moe_cfg.router_bias is True
    assert moe_cfg.mlp_bias is True
    assert moe_cfg.topk_method == 'noaux_tc'
    assert moe_cfg.scoring_func == 'sigmoid'
    assert moe_cfg.routed_scale == 2.0
    assert moe_cfg.topk_group == 2
    assert moe_cfg.n_group == 2
    assert moe_cfg.router_n_groups == 4
    assert moe_cfg.hidden_dim == 32
    assert moe_cfg.data_type == _tm.DataType.TYPE_BF16
    assert moe_cfg.tp_size == 2

def _engine_cfg():
    return SimpleNamespace(
        rope_scaling_factor=0,
        attn_tp_size=2,
        attn_cp_size=1,
        mlp_tp_size=4,
    )


def _resolver():
    return SimpleNamespace(data_type=_tm.DataType.TYPE_FP16)


def test_llama_constructor_preserves_common_config_fields():
    from lmdeploy.turbomind.models.llama import LlamaModel

    cfg = DummyConfig(
        num_hidden_layers=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        model_type='llama',
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        head_dim=8,
        max_position_embeddings=1024,
        intermediate_size=64,
        attention_bias=False,
    )

    model = LlamaModel(cfg, _engine_cfg(), resolver=_resolver())

    assert model.cfg is cfg
    assert model.cfg.num_hidden_layers == 2
    assert model.cfg.vocab_size == 128
    assert model.cfg.rms_norm_eps == 1e-6
    assert model._attn_cfg.hidden_dim == 32
    assert model._attn_cfg.head_num == 4
    assert model._attn_cfg.kv_head_num == 2
    assert not hasattr(model, '_head_dim')
    assert not hasattr(model, '_head_num')
    assert not hasattr(model, '_kv_head_num')
    assert not hasattr(model, '_rope')
    assert model._attn_cfg.has_bias is False
    assert model._ffn_cfg.inter_size == 64
    assert model._ffn_cfg.tp_size == 4


def test_qwen2_constructor_keeps_qkv_bias_local():
    from lmdeploy.turbomind.models.qwen2 import Qwen2Model

    cfg = DummyConfig(
        num_hidden_layers=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        model_type='qwen2',
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        head_dim=8,
        max_position_embeddings=1024,
        intermediate_size=64,
        qkv_bias=True,
    )

    model = Qwen2Model(cfg, _engine_cfg(), resolver=_resolver())

    assert model._attn_cfg.has_bias is True
    assert model._ffn_cfg.inter_size == 64


def test_gpt_oss_constructor_keeps_sliding_window_local():
    from lmdeploy.turbomind.models.gpt_oss import GptOssModel

    cfg = DummyConfig(
        num_hidden_layers=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        model_type='gpt-oss',
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        head_dim=8,
        max_position_embeddings=1024,
        intermediate_size=64,
        attention_bias=True,
        num_local_experts=4,
        experts_per_token=2,
        layer_types=['sliding_attention', 'full_attention'],
        sliding_window=256,
    )

    model = GptOssModel(cfg, _engine_cfg(), resolver=_resolver())

    assert model._attn_cfg.attn_sink is True
    assert model._attn_cfg.has_bias == 1
    assert model._window_sizes == [256, 0]
    assert model._expert_nums == [4, 4]
