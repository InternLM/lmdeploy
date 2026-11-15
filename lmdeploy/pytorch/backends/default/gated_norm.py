# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
    # TODO: used custom kernel
    from fla.modules import FusedRMSNormGated
    try:
        # avoid unwanted specialize
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel
        keys = layer_norm_gated_fwd_kernel.fn.keys
        if 'NB' in keys:
            keys.remove('NB')
    except Exception:
        logger.debug('patch layer_norm_gated_fwd_kernel autotuning failed.')
    return FusedRMSNormGated(hidden_size, eps=eps, **kwargs)
