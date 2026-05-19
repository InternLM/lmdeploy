# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind."""
from __future__ import annotations

from ._base import Builder, BuiltModule, SplitSide, _act_type_id, _torch_dtype_to_cpp
from .attention import AttentionBuilder
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .deltanet import DeltaNetBuilder
from .ffn import FfnBuilder, fuse_w1w3
from .mla import MLABuilder
from .module_list import ModuleListBuilder, ModuleListConfig
from .moe import MoeBuilder
from .norm import NormBuilder, make_norm_config
from .text_model import TextModelBuilder

__all__ = [
    # Base
    'Builder', 'BuiltModule', 'TextModelBuilder', 'SplitSide',
    '_act_type_id', '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder', 'FfnBuilder', 'MoeBuilder',
    'DeltaNetBuilder', 'MLABuilder',
    'DecoderLayerBuilder', 'ModuleListBuilder',
    'NormBuilder',
    # Primitive config wrappers
    'make_norm_config',
    # C++ config re-exports
    'DecoderLayerConfig', 'ModuleListConfig',
    # Helper functions
    'fuse_w1w3',
]
