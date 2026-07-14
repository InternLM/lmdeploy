# Copyright (c) OpenMMLab. All rights reserved.
"""Source-model registry.

The INPUT_MODELS registry maps an architecture name to its TextModel
subclass. Models register themselves via ``@INPUT_MODELS.register_module(name=...)``.
"""
from __future__ import annotations

from mmengine import Registry

INPUT_MODELS = Registry('source model',
                        locations=['lmdeploy.turbomind.models.base'])
