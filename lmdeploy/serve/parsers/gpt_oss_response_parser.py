# Copyright (c) OpenMMLab. All rights reserved.
"""GPT-OSS response parser entry; loads Harmony implementation only when
openai_harmony is installed."""
from __future__ import annotations

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_OPENAI_HARMONY_AVAILABLE = False
try:
    import openai_harmony  # noqa: F401
except ImportError as e:
    logger.warning(
        'openai_harmony import failed (%s). Install openai_harmony for GPT-OSS Harmony response '
        'parsing; without it the server uses the default response parser for GPT-OSS models.',
        e,
    )
else:
    _OPENAI_HARMONY_AVAILABLE = True

GptOssResponseParser = None  # type: ignore[misc, assignment]
if _OPENAI_HARMONY_AVAILABLE:
    from ._openai_harmony import GptOssResponseParser  # noqa: F401
