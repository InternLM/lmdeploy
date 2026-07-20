# Copyright (c) OpenMMLab. All rights reserved.

import torch
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
from lmdeploy.pytorch.configurations.hy3 import (
    Hy3ModelConfigBuilder,
)
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.hy3 import Hy3MLP, Hy3MoE
from lmdeploy.pytorch.models.patch import (
    build_model_context,
    build_model_from_hf_config,
)


def _make_hy3_config(num_hidden_layers=2):
    return HYV3Config(
        architectures=['HYV3ForCausalLM'],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
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


def test_hy3_config_builder_is_registered():
    config = _make_hy3_config()

    assert Hy3ModelConfigBuilder in (
        AutoModelConfigBuilder._sub_classes
    )
    assert Hy3ModelConfigBuilder.condition(config)


def test_hy3_factory_builds_dense_and_moe_layers():
    config = _make_hy3_config()

    model = build_model_from_hf_config(
        config,
        dtype=torch.bfloat16,
        device=torch.device('meta'),
    )

    assert type(model).__name__ == 'HYV3ForCausalLM'
    assert type(model).__module__ == 'lmdeploy.pytorch.models.hy3'
    assert len(model.model.layers) == 2
    assert isinstance(model.model.layers[0].mlp, Hy3MLP)
    assert isinstance(model.model.layers[1].mlp, Hy3MoE)
    assert model.model.embed_tokens.weight.shape == (32, 16)
    assert model.lm_head.weight.shape == (32, 16)

def test_hy3_loads_qkv_weights():
    config = _make_hy3_config(num_hidden_layers=1)
    model = build_model_from_hf_config(
        config,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    q_weight = torch.arange(
        16 * 16,
        dtype=torch.float32,
    ).reshape(16, 16)
    k_weight = torch.arange(
        8 * 16,
        dtype=torch.float32,
    ).reshape(8, 16) + 1000
    v_weight = torch.arange(
        8 * 16,
        dtype=torch.float32,
    ).reshape(8, 16) + 2000

    model.load_weights([
        (
            'model.layers.0.self_attn.q_proj.weight',
            q_weight,
        ),
        (
            'model.layers.0.self_attn.k_proj.weight',
            k_weight,
        ),
        (
            'model.layers.0.self_attn.v_proj.weight',
            v_weight,
        ),
    ])

    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    expected_weight = torch.cat(
        [q_weight, k_weight, v_weight],
        dim=0,
    )

    torch.testing.assert_close(
        qkv_proj.weight,
        expected_weight,
    )

    hidden_states = torch.arange(
        2 * 16,
        dtype=torch.float32,
    ).reshape(2, 16)

    qkv_states = qkv_proj(hidden_states)
    query_states, key_states, value_states = (
        qkv_proj.split_qkv(qkv_states)
    )

    torch.testing.assert_close(
        query_states.reshape(2, -1),
        torch.nn.functional.linear(
            hidden_states,
            q_weight,
        ),
    )
    torch.testing.assert_close(
        key_states.reshape(2, -1),
        torch.nn.functional.linear(
            hidden_states,
            k_weight,
        ),
    )
    torch.testing.assert_close(
        value_states.reshape(2, -1),
        torch.nn.functional.linear(
            hidden_states,
            v_weight,
        ),
    )

def test_hy3_loads_dense_mlp_weights():
    config = _make_hy3_config(num_hidden_layers=1)
    model = build_model_from_hf_config(
        config,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    gate_weight = torch.arange(
        32 * 16,
        dtype=torch.float32,
    ).reshape(32, 16)
    up_weight = torch.arange(
        32 * 16,
        dtype=torch.float32,
    ).reshape(32, 16) + 1000
    down_weight = torch.arange(
        16 * 32,
        dtype=torch.float32,
    ).reshape(16, 32) + 2000

    model.load_weights([
        (
            'model.layers.0.mlp.gate_proj.weight',
            gate_weight,
        ),
        (
            'model.layers.0.mlp.up_proj.weight',
            up_weight,
        ),
        (
            'model.layers.0.mlp.down_proj.weight',
            down_weight,
        ),
    ])

    mlp = model.model.layers[0].mlp

    torch.testing.assert_close(
        mlp.gate_up_proj.weight,
        torch.cat(
            [gate_weight, up_weight],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        mlp.down_proj.weight,
        down_weight,
    )

def test_hy3_loads_expert_weights():
    config = _make_hy3_config()
    model = build_model_from_hf_config(
        config,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    weights = []
    expected_weights = []

    for expert_id in range(config.num_experts):
        gate_weight = torch.full(
            (8, 16),
            float(expert_id + 1),
        )
        up_weight = torch.full(
            (8, 16),
            float(expert_id + 11),
        )
        down_weight = torch.full(
            (16, 8),
            float(expert_id + 21),
        )

        prefix = (
            f'model.layers.1.mlp.experts.{expert_id}'
        )
        weights.extend([
            (
                f'{prefix}.gate_proj.weight',
                gate_weight,
            ),
            (
                f'{prefix}.up_proj.weight',
                up_weight,
            ),
            (
                f'{prefix}.down_proj.weight',
                down_weight,
            ),
        ])
        expected_weights.append((
            torch.cat(
                [gate_weight, up_weight],
                dim=0,
            ),
            down_weight,
        ))

    model.load_weights(weights)

    experts = model.model.layers[1].mlp.experts

    for expert_id, (
        expected_gate_up,
        expected_down,
    ) in enumerate(expected_weights):
        torch.testing.assert_close(
            experts.gate_up.weight[expert_id],
            expected_gate_up,
        )
        torch.testing.assert_close(
            experts.down.weight[expert_id],
            expected_down,
        )

def test_hy3_loads_router_and_shared_mlp_weights():
    config = _make_hy3_config()
    model = build_model_from_hf_config(
        config,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    router_weight = torch.arange(
        4 * 16,
        dtype=torch.float32,
    ).reshape(4, 16)
    expert_bias = torch.arange(
        4,
        dtype=torch.float32,
    )
    shared_gate_weight = torch.full(
        (8, 16),
        1.0,
    )
    shared_up_weight = torch.full(
        (8, 16),
        2.0,
    )
    shared_down_weight = torch.full(
        (16, 8),
        3.0,
    )

    model.load_weights([
        (
            'model.layers.1.mlp.router.gate.weight',
            router_weight,
        ),
        (
            'model.layers.1.mlp.expert_bias',
            expert_bias,
        ),
        (
            'model.layers.1.mlp.shared_mlp'
            '.gate_proj.weight',
            shared_gate_weight,
        ),
        (
            'model.layers.1.mlp.shared_mlp'
            '.up_proj.weight',
            shared_up_weight,
        ),
        (
            'model.layers.1.mlp.shared_mlp'
            '.down_proj.weight',
            shared_down_weight,
        ),
    ])

    moe = model.model.layers[1].mlp

    torch.testing.assert_close(
        moe.router.gate.weight,
        router_weight,
    )
    torch.testing.assert_close(
        moe.expert_bias,
        expert_bias,
    )
    torch.testing.assert_close(
        moe.shared_mlp.gate_up_proj.weight,
        torch.cat(
            [shared_gate_weight, shared_up_weight],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        moe.shared_mlp.down_proj.weight,
        shared_down_weight,
    )

def test_hy3_skips_mtp_weights():
    config = _make_hy3_config()
    model = build_model_from_hf_config(
        config,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    norm_weight = torch.arange(
        config.hidden_size,
        dtype=torch.float32,
    )

    model.load_weights([
        (
            # Tiny config has two main layers, so layer 2
            # represents the first MTP layer.
            'model.layers.2.self_attn.q_proj.weight',
            torch.empty(1),
        ),
        (
            'model.norm.weight',
            norm_weight,
        ),
    ])

    torch.testing.assert_close(
        model.model.norm.weight,
        norm_weight,
    )


def test_hy3_static_fp8_factory_builds_static_modules():
    config = _make_hy3_config()

    config.quantization_config = {
        'quant_method': 'fp8',
        'activation_scheme': 'static',
        'ignored_layers': [
            'lm_head',
            'model.embed_tokens',
        ],
    }

    quant_config = QuantizationConfig.from_config(
        config,
    )

    context = BuildModelContext(
        quant_config=quant_config,
    )

    with build_model_context(context):
        model = build_model_from_hf_config(
            config,
            dtype=torch.bfloat16,
            device=torch.device('meta'),
        )

    dense_layer = model.model.layers[0]
    moe_layer = model.model.layers[1]

    assert type(
        dense_layer.self_attn.qkv_proj
    ).__name__ == 'QKVStaticF8Linear'

    assert type(
        dense_layer.mlp.gate_up_proj
    ).__name__ == 'MergedStaticF8Linear'

    assert type(
        moe_layer.mlp.experts
    ).__name__ == 'FusedMoEStaticF8'

    assert type(
        moe_layer.mlp.shared_mlp.gate_up_proj
    ).__name__ == 'MergedStaticF8Linear'

    experts = moe_layer.mlp.experts

    assert experts.gate_up.weight.shape == (
        4,
        16,
        16,
    )
    assert experts.gate_up.weight.dtype == (
        torch.float8_e4m3fn
    )
    assert experts.gate_up.weight_scale.shape == (
        4,
        16,
        1,
    )
    assert experts.gate_up.input_scale.shape == (1,)

    assert experts.down.weight.shape == (
        4,
        16,
        8,
    )
    assert experts.down.weight.dtype == (
        torch.float8_e4m3fn
    )
    assert experts.down.weight_scale.shape == (
        4,
        16,
        1,
    )
    assert experts.down.input_scale.shape == (1,)

def test_hy3_loads_static_fp8_expert_weights():
    config = _make_hy3_config()

    config.quantization_config = {
        'quant_method': 'fp8',
        'activation_scheme': 'static',
        'ignored_layers': [
            'lm_head',
            'model.embed_tokens',
        ],
    }

    quant_config = QuantizationConfig.from_config(
        config,
    )

    context = BuildModelContext(
        quant_config=quant_config,
    )

    with build_model_context(context):
        model = build_model_from_hf_config(
            config,
            dtype=torch.bfloat16,
            device=torch.device('cpu'),
        )

    ffn_dim = config.moe_intermediate_size
    hidden_dim = config.hidden_size

    gate_weight = torch.arange(
        ffn_dim * hidden_dim,
        dtype=torch.float32,
    ).reshape(
        ffn_dim,
        hidden_dim,
    ).to(torch.float8_e4m3fn)

    up_weight = (
        torch.arange(
            ffn_dim * hidden_dim,
            dtype=torch.float32,
        ).reshape(
            ffn_dim,
            hidden_dim,
        )
        + 128
    ).to(torch.float8_e4m3fn)

    down_weight = (
        torch.arange(
            hidden_dim * ffn_dim,
            dtype=torch.float32,
        ).reshape(
            hidden_dim,
            ffn_dim,
        )
        + 256
    ).to(torch.float8_e4m3fn)

    gate_weight_scale = torch.tensor(
        0.001,
        dtype=torch.bfloat16,
    )
    up_weight_scale = torch.tensor(
        0.002,
        dtype=torch.bfloat16,
    )
    down_weight_scale = torch.tensor(
        0.003,
        dtype=torch.bfloat16,
    )

    gate_up_input_scale = torch.tensor(
        [0.004],
        dtype=torch.float32,
    )
    down_input_scale = torch.tensor(
        [0.005],
        dtype=torch.float32,
    )

    prefix = 'model.layers.1.mlp.experts.0'

    model.load_weights([
        (
            f'{prefix}.gate_proj.weight',
            gate_weight,
        ),
        (
            f'{prefix}.gate_proj.weight_scale',
            gate_weight_scale,
        ),
        (
            f'{prefix}.gate_proj.input_scale',
            gate_up_input_scale,
        ),
        (
            f'{prefix}.up_proj.weight',
            up_weight,
        ),
        (
            f'{prefix}.up_proj.weight_scale',
            up_weight_scale,
        ),
        (
            f'{prefix}.up_proj.input_scale',
            gate_up_input_scale,
        ),
        (
            f'{prefix}.down_proj.weight',
            down_weight,
        ),
        (
            f'{prefix}.down_proj.weight_scale',
            down_weight_scale,
        ),
        (
            f'{prefix}.down_proj.input_scale',
            down_input_scale,
        ),
    ])

    experts = model.model.layers[1].mlp.experts

    assert torch.equal(
        experts.gate_up.weight[
            0,
            :ffn_dim,
        ],
        gate_weight,
    )

    assert torch.equal(
        experts.gate_up.weight[
            0,
            ffn_dim:,
        ],
        up_weight,
    )

    assert torch.equal(
        experts.down.weight[0],
        down_weight,
    )

    torch.testing.assert_close(
        experts.gate_up.weight_scale[
            0,
            :ffn_dim,
        ],
        torch.full(
            (ffn_dim, 1),
            gate_weight_scale.float().item(),
        ),
    )

    torch.testing.assert_close(
        experts.gate_up.weight_scale[
            0,
            ffn_dim:,
        ],
        torch.full(
            (ffn_dim, 1),
            up_weight_scale.float().item(),
        ),
    )

    torch.testing.assert_close(
        experts.down.weight_scale[0],
        torch.full(
            (hidden_dim, 1),
            down_weight_scale.float().item(),
        ),
    )

    torch.testing.assert_close(
        experts.gate_up.input_scale,
        gate_up_input_scale,
        atol=0,
        rtol=0,
    )

    torch.testing.assert_close(
        experts.down.input_scale,
        down_input_scale,
        atol=0,
        rtol=0,
    )
