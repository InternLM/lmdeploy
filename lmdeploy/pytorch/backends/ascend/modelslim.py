# Copyright (c) 2024, DeepLink. All rights reserved.
"""ModelSlim metadata and checkpoint adapters for the Ascend backend."""


def get_modelslim_quant_method(self, prefix, module_kind):
    if not prefix or module_kind == 'norm':
        return None

    description = self.hf_quant_config.get('quant_description', {})
    if not description:
        raise ValueError('ModelSlim quantization requires quant_description metadata.')

    proj_name = prefix.rsplit('.', 1)[-1]
    if module_kind == 'moe':
        suffixes = ('0.gate_proj.weight', '0.up_proj.weight', '0.down_proj.weight')
        keys = [f"{prefix}.{suffix}" for suffix in suffixes]
    elif proj_name == 'gate_up_proj':
        parent = prefix.rsplit('.', 1)[0]
        keys = [f"{parent}.gate_proj.weight", f"{parent}.up_proj.weight"]
    else:
        keys = [f"{prefix}.weight"]

    missing = [key for key in keys if key not in description]
    if missing:
        return None
    quant_types = {description[key] for key in keys}
    if len(quant_types) != 1:
        raise ValueError(f"ModelSlim fused module {prefix} mixes quant types: {sorted(quant_types)}")

    quant_type = quant_types.pop()
    if quant_type == 'FLOAT':
        return None
    if quant_type == 'W8A8_DYNAMIC':
        return 'smooth_quant'
    if quant_type == 'W8A8':
        if module_kind == 'moe':
            raise ValueError(f"Static W8A8 MoE is not supported for {prefix}.")
        return 'modelslim_w8a8_static'
    raise ValueError(f"Unsupported ModelSlim quant type {quant_type!r} for {prefix}.")


def map_modelslim_param_name(config, name, params_dict):
    quantization_config = getattr(config, 'quantization_config', None) or {}
    if quantization_config.get('quant_method') != 'modelslim':
        return name
    if name.endswith('.weight_offset'):
        return None
    if name.endswith('.weight_scale'):
        mapped_name = name.removesuffix('.weight_scale') + '.scale'
        return mapped_name if mapped_name in params_dict else None
    return name


def convert_modelslim_weights(self, weights):
    quantization_config = getattr(self.config, 'quantization_config', None) or {}
    if quantization_config.get('quant_method') != 'modelslim':
        return weights

    params_dict = dict(self.named_parameters())
    stacked_params_mapping = [
        ('.gate_up_proj', '.gate_proj'),
        ('.gate_up_proj', '.up_proj'),
    ]
    if not getattr(self.config, 'use_mla', True):
        stacked_params_mapping.extend(
            [
                ('.qkv_proj', '.q_proj'),
                ('.qkv_proj', '.k_proj'),
                ('.qkv_proj', '.v_proj'),
            ]
        )

    def convert_weights():
        for name, loaded_weight in weights:
            is_attention = '.self_attn' in name and getattr(self.config, 'use_mla', True)
            if '.experts' in name or is_attention:
                yield name, loaded_weight
                continue
            if name.endswith('.weight_offset'):
                continue
            if name.endswith('.weight_scale'):
                mapped_name = name.removesuffix('.weight_scale') + '.scale'
                param_name = mapped_name
                for fused_name, shard_name in stacked_params_mapping:
                    if shard_name in param_name:
                        param_name = param_name.replace(shard_name, fused_name)
                        break
                if param_name not in params_dict:
                    continue
                name = mapped_name
            yield name, loaded_weight

    return convert_weights()
