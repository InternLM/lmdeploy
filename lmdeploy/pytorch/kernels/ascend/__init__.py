# Copyright (c) OpenMMLab. All rights reserved.
"""Native Ascend kernels, loaded when an operator is constructed or invoked."""

from importlib import import_module

_EXPORTS = {
    'multinomial_sampling': ('..default', 'multinomial_sampling'),
    'per_channel_quant': ('..default', 'per_channel_quant'),
    'apply_rotary_pos_emb': ('.apply_rotary_pos_emb', 'apply_rotary_pos_emb'),
    'apply_rotary_pos_emb_interleaved': ('.apply_rotary_pos_emb', 'apply_rotary_pos_emb_interleaved'),
    'awq_linear': ('.awq_kernels', 'awq_linear'),
    'fill_kv_cache': ('.fill_kv_cache', 'fill_kv_cache'),
    'flash_attention_fwd': ('.flash_attention', 'flash_attention_fwd'),
    'AscendMoECommType': ('.fused_moe', 'AscendMoECommType'),
    'AscendMoeMetadata': ('.fused_moe', 'AscendMoeMetadata'),
    'fused_moe': ('.fused_moe', 'fused_moe'),
    'fused_moe_w8a8': ('.fused_moe', 'fused_moe_w8a8'),
    'lightning_indexer': ('.lightning_indexer', 'lightning_indexer'),
    'linear': ('.linear', 'linear'),
    'moe_gating_topk_softmax': ('.moe_gating_topk_softmax', 'moe_gating_topk_softmax'),
    'paged_attention_fwd': ('.pagedattention', 'paged_attention_fwd'),
    'rms_norm': ('.rms_norm', 'rms_norm'),
    'sparse_attention_fwd': ('.sparse_attention', 'sparse_attention_fwd'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
