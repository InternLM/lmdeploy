# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import _turbomind as _tm

from lmdeploy.turbomind.models.base import INPUT_MODELS
from lmdeploy.turbomind.models.muse_glimmer import (
    MuseGlimmerTextModel,
    MuseGlimmerVisionModel,
    _config_namespace,
)


def _text_config():
    return SimpleNamespace(
        vocab_size=202048,
        hidden_size=6656,
        intermediate_size=19968,
        num_hidden_layers=52,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        sliding_window=2048,
        final_logit_softcapping=20.0,
        output_multiplier=0.19611613513818404,
        qk_scale_factor=3.87,
        rope_parameters={'rope_type': 'default', 'rope_theta': 500000.0},
        layer_types=['sliding_attention'] * 51 + ['full_attention'],
        layer_rope_theta=[500000.0] * 51 + [0],
    )


def _vision_config():
    return SimpleNamespace(
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=50,
        num_attention_heads=16,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=32,
        pos_emb_width=32,
        layer_norm_eps=1e-5,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0},
        layer_types=[
            'full_attention' if (index + 1) % 4 == 0 or index == 49 else 'window_attention'
            for index in range(50)
        ],
    )


def test_text_runtime_configuration():
    model = MuseGlimmerTextModel(_text_config(), resolver=object())

    assert model._attn_cfg.head_dim == 128
    assert model._attn_cfg.head_num == 32
    assert model._attn_cfg.kv_head_num == 2
    assert model._attn_cfg.output_gate
    assert model._attn_cfg.rope.base == 500000.0
    assert model._ffn_cfg.inter_size == 19968


def test_vision_runtime_configuration():
    text_cfg = _text_config()
    vision_cfg = _vision_config()
    config = SimpleNamespace(text_config=text_cfg, vision_config=vision_cfg)
    resolver = SimpleNamespace(data_type=_tm.DataType.TYPE_BF16)
    model = MuseGlimmerVisionModel(vision_cfg, config, resolver=resolver)
    runtime = model._make_vision_root_cfg()

    assert runtime.hidden_dim == 1536
    assert runtime.out_hidden_dim == 6656
    assert runtime.depth == 50
    assert runtime.output_spatial_merge_size == 2
    assert runtime.zero_padded_pos_embed
    assert runtime.rope_axes_w_first
    assert runtime.rope_position_offset == 1
    assert runtime.rope_theta == 10000.0
    assert runtime.pixel_shuffle
    assert runtime.merger_double_gelu
    assert runtime.fullatt_block_indexes[-1] == 49


def test_nested_config_dict_conversion_and_registration():
    config = _config_namespace({
        'text_config': {'hidden_size': 6656},
        'id2label': {0: 'LABEL_0'},
    })

    assert config.text_config.hidden_size == 6656
    assert config.id2label == {0: 'LABEL_0'}
    assert INPUT_MODELS.get('muse_glimmer').__name__ == 'MuseGlimmerModel'
