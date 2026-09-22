# Copyright (c) OpenMMLab. All rights reserved.
"""``/v1/chat/completions`` endpoint package.

Exposes ``register`` for router assembly. Chat/shared Pydantic models live in
the top-level :mod:`lmdeploy.serve.openai.protocol` module (they are shared
across the codebase), so this package only contains the serving, validation,
logprobs and logits-processor logic.
"""
from __future__ import annotations

from .serving import register

__all__ = ['register']
