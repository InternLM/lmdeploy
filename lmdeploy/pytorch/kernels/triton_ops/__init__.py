from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
from .fla.chunk import chunk_gated_delta_rule
from .fla.sigmoid_gating import fused_recurrent_gated_delta_rule
from .sigmoid_gating_delta_rule import fused_sigmoid_gating_delta_rule_update
from .rms_norm_gated import RMSNormGated

__all__ = [
    'causal_conv1d_fn',
    'causal_conv1d_update_npu',
    'chunk_gated_delta_rule',
    'fused_recurrent_gated_delta_rule',
    'fused_sigmoid_gating_delta_rule_update',   # 针对decoding极致的优化
    'RMSNormGated'
]
