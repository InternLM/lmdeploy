# Copyright (c) OpenMMLab. All rights reserved.
from .cal_qparams import (cal_qparams_per_channel_absmax,
                          cal_qparams_per_channel_minmax,
                          cal_qparams_per_group_absmax,
                          cal_qparams_per_group_minmax,
                          cal_qparams_per_tensor_absmax,
                          cal_qparams_per_tensor_minmax)
from .calib_dataloader import get_calib_loaders
from .collect import collect_target_modules, collect_target_weights
from .memory_efficient import memory_efficient_inference

__all__ = [
    'cal_qparams_per_channel_absmax', 'cal_qparams_per_channel_minmax',
    'cal_qparams_per_group_absmax', 'cal_qparams_per_group_minmax',
    'cal_qparams_per_tensor_absmax', 'cal_qparams_per_tensor_minmax',
    'get_calib_loaders', 'memory_efficient_inference',
    'collect_target_modules', 'collect_target_weights'
]
