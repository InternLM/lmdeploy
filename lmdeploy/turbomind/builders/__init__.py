# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind."""

from __future__ import annotations

from ._base import Builder, BuiltModule, Context, SplitSide, _act_type_id, _torch_dtype_to_cpp
from .attention import AttentionBuilder
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .deltanet import DeltaNetBuilder
from .eagle3_weight import Eagle3WeightBuilder, Eagle3WeightConfig
from .ffn import FfnBuilder
from .mla import MLABuilder
from .module_list import ModuleListBuilder, ModuleListConfig
from .moe import MoeBuilder
from .norm import LayerNormBuilder, NormBuilder, make_layer_norm_config, make_norm_config
from .qwen3_5_mtp_weight import Qwen35MtpWeightBuilder, Qwen35MtpWeightConfig
from .text_model import TextModelBuilder
from .vision_model import VisionModelBuilder

__all__ = [
    # Base
    'Builder',
    'BuiltModule',
    'Context',
    'TextModelBuilder',
    'VisionModelBuilder',
    'SplitSide',
    '_act_type_id',
    '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder',
    'FfnBuilder',
    'MoeBuilder',
    'DeltaNetBuilder',
    'MLABuilder',
    'DecoderLayerBuilder',
    'Eagle3WeightBuilder',
    'Qwen35MtpWeightBuilder',
    'ModuleListBuilder',
    'NormBuilder',
    'LayerNormBuilder',
    # Primitive config wrappers
    'make_norm_config',
    'make_layer_norm_config',
    # C++ config re-exports
    'DecoderLayerConfig',
    'Eagle3WeightConfig',
    'Qwen35MtpWeightConfig',
    'ModuleListConfig',
    # Helper functions
]
