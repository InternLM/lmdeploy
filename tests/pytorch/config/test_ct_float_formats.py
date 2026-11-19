# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Parse and dispatch of the compressed-tensors float lanes.

Covers the two new formats (`nvfp4-pack-quantized`, `float-quantized`) plus the unchanged historical `pack-quantized`
behavior, the quant-method dispatch matrix, and the load-time normalization guard (float lanes keep their raw checkpoint
suffixes).
"""
import pytest

from lmdeploy.pytorch.config import QuantizationConfig, _parse_compressed_tensors_config

_NVFP4_CONFIG = {
    'quant_method': 'compressed-tensors',
    'format': 'nvfp4-pack-quantized',
    'quantization_status': 'compressed',
    'version': '0.17.0',
    'producer': {
        'name': 'lossaware-quant',
        'version': '0.0.0'
    },
    'kv_cache_scheme': None,
    'global_compression_ratio': None,
    'config_groups': {
        'group_0': {
            'format': 'nvfp4-pack-quantized',
            'targets': ['Linear'],
            'input_activations': None,
            'output_activations': None,
            'weights': {
                'num_bits': 4,
                'group_size': 16,
                'strategy': 'tensor_group',
                'symmetric': True,
                'dynamic': False,
                'type': 'float',
                'observer': 'minmax',
            },
        }
    },
    'ignore': ['lm_head', 're:.*visual.*'],
}

_FP8_PER_CHANNEL_CONFIG = {
    'quant_method': 'compressed-tensors',
    'format': 'float-quantized',
    'quantization_status': 'compressed',
    'sparsity_config': {},
    'transform_config': {},
    'config_groups': {
        'group_0': {
            'format': 'float-quantized',
            'targets': ['Linear'],
            'input_activations': {
                'num_bits': 8,
                'type': 'float',
                'strategy': 'token',
                'dynamic': True,
                'symmetric': True,
                'group_size': None,
            },
            'output_activations': None,
            'weights': {
                'num_bits': 8,
                'group_size': None,
                'strategy': 'channel',
                'symmetric': True,
                'dynamic': False,
                'type': 'float',
            },
        }
    },
    'ignore': ['lm_head'],
}

_PACK_QUANTIZED_CONFIG = {
    'quant_method': 'compressed-tensors',
    'format': 'pack-quantized',
    'quantization_status': 'compressed',
    'config_groups': {
        'group_0': {
            'targets': ['Linear'],
            'input_activations': None,
            'output_activations': None,
            'weights': {
                'num_bits': 4,
                'group_size': 128,
                'strategy': 'group',
                'symmetric': True,
                'dynamic': False,
                'type': 'int',
            },
        }
    },
    'ignore': [],
}


def _from(quant_config):
    """Build a QuantizationConfig the way ModelConfig does."""

    class _Hf:

        @property
        def quantization_config(self):
            return quant_config

    return QuantizationConfig.from_config(_Hf())


class TestParse:

    def test_nvfp4(self):
        bits, group_size, ignored, ct_format = _parse_compressed_tensors_config(_NVFP4_CONFIG)
        assert (bits, group_size, ct_format) == (4, 16, 'nvfp4-pack-quantized')
        assert ignored == ['lm_head', 're:.*visual.*']

    def test_fp8_per_channel(self):
        bits, group_size, ignored, ct_format = _parse_compressed_tensors_config(_FP8_PER_CHANNEL_CONFIG)
        assert (bits, group_size, ct_format) == (8, None, 'float-quantized')
        assert ignored == ['lm_head']

    def test_fp8_blocked(self):
        config = dict(_FP8_PER_CHANNEL_CONFIG)
        config['config_groups'] = {
            'group_0': {
                'format': 'float-quantized',
                'targets': ['Linear'],
                'input_activations': None,
                'output_activations': None,
                'weights': {
                    'num_bits': 8,
                    'group_size': 128,
                    'strategy': 'tensor_group',
                    'symmetric': True,
                    'dynamic': False,
                    'type': 'float',
                },
            }
        }
        bits, group_size, _, ct_format = _parse_compressed_tensors_config(config)
        assert (bits, group_size, ct_format) == (8, 128, 'float-quantized')

    def test_pack_quantized_unchanged(self):
        bits, group_size, ignored, ct_format = _parse_compressed_tensors_config(_PACK_QUANTIZED_CONFIG)
        assert (bits, group_size, ct_format) == (4, 128, 'pack-quantized')
        assert ignored == []

    def test_unknown_format_rejected(self):
        config = dict(_NVFP4_CONFIG, format='mystery')
        with pytest.raises(ValueError, match='Unsupported compressed-tensors `format`'):
            _parse_compressed_tensors_config(config)

    def test_nvfp4_rejects_wrong_group(self):
        config = dict(_NVFP4_CONFIG)
        config['config_groups'] = {
            'group_0': dict(_NVFP4_CONFIG['config_groups']['group_0'],
                            weights=dict(_NVFP4_CONFIG['config_groups']['group_0']['weights'], group_size=32))
        }
        with pytest.raises(ValueError, match='group_size'):
            _parse_compressed_tensors_config(config)

    def test_float_rejects_static_input_activations(self):
        config = dict(_FP8_PER_CHANNEL_CONFIG)
        config['config_groups'] = {
            'group_0': dict(_FP8_PER_CHANNEL_CONFIG['config_groups']['group_0'],
                            input_activations={
                                'num_bits': 8,
                                'type': 'float',
                                'strategy': 'tensor',
                                'dynamic': False,
                            })
        }
        with pytest.raises(ValueError, match='static per-tensor W8A8 input scales are unsupported'):
            _parse_compressed_tensors_config(config)

    def test_float_rejects_int_weights(self):
        config = dict(_FP8_PER_CHANNEL_CONFIG)
        config['config_groups'] = {
            'group_0': dict(_FP8_PER_CHANNEL_CONFIG['config_groups']['group_0'],
                            weights=dict(_FP8_PER_CHANNEL_CONFIG['config_groups']['group_0']['weights'],
                                         num_bits=4))
        }
        with pytest.raises(ValueError, match='num_bits'):
            _parse_compressed_tensors_config(config)

    def test_tolerates_unknown_top_level_keys(self):
        # sparsity_config / transform_config / kv_cache_scheme / producer
        # must not disturb the parse.
        config = dict(_FP8_PER_CHANNEL_CONFIG, unexpected_key={'a': 1})
        assert _parse_compressed_tensors_config(config)[2] == ['lm_head']


class TestDispatch:

    def test_nvfp4_dispatch(self):
        qc = _from(_NVFP4_CONFIG)
        assert qc.quant_method == 'compressed-tensors'
        assert qc.get_quant_method('model.layers.3.mlp.gate_up_proj') == 'nvfp4'
        assert qc.get_quant_method('model.layers.3.mlp.down_proj') == 'nvfp4'
        assert qc.get_quant_method('model.layers.3.mlp.gate_up_proj', module_kind='moe') == 'nvfp4'

    def test_ct_fp8_dispatch(self):
        qc = _from(_FP8_PER_CHANNEL_CONFIG)
        assert qc.get_quant_method('model.layers.0.self_attn.qkv_proj') == 'ct_fp8'
        assert qc.get_quant_method('model.layers.0.self_attn.qkv_proj', module_kind='moe') == 'ct_fp8'

    def test_norm_dispatches_none(self):
        assert _from(_NVFP4_CONFIG).get_quant_method('model.layers.0.input_layernorm',
                                                     module_kind='norm') is None
        assert _from(_FP8_PER_CHANNEL_CONFIG).get_quant_method('model.layers.0.input_layernorm',
                                                               module_kind='norm') is None

    def test_ignore_rules(self):
        qc = _from(_NVFP4_CONFIG)
        assert qc.get_quant_method('lm_head') is None
        assert qc.get_quant_method('model.visual.blocks.0.attn.qkv') is None
        assert qc.get_quant_method('model.layers.0.mlp.down_proj') == 'nvfp4'
        qc8 = _from(_FP8_PER_CHANNEL_CONFIG)
        assert qc8.get_quant_method('lm_head') is None
        assert qc8.get_quant_method('model.layers.0.mlp.down_proj') == 'ct_fp8'

    def test_group128_int4_still_awq(self):
        qc = _from(_PACK_QUANTIZED_CONFIG)
        assert qc.ct_format == 'pack-quantized'
        assert qc.get_quant_method('model.layers.0.mlp.down_proj') == 'awq'


class TestLoadNormalize:

    def test_float_lanes_keep_raw_suffixes(self):
        assert _from(_NVFP4_CONFIG).load_normalize() is False
        assert _from(_FP8_PER_CHANNEL_CONFIG).load_normalize() is False
        assert _from(_PACK_QUANTIZED_CONFIG).load_normalize() is True
