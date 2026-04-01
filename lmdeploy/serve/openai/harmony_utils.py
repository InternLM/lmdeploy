# Copyright (c) OpenMMLab. All rights reserved.
"""Backward-compatibility shim for GPT-OSS Harmony parser.

The canonical implementation now lives in:
`lmdeploy.serve.openai.reasoning_parser.gpt_oss_reasoning_parser`.
"""

from .reasoning_parser.gpt_oss_reasoning_parser import (  # noqa: F401
    GptOssChatParser,
    get_encoding,
    get_streamable_parser_for_assistant,
)

__all__ = ['GptOssChatParser', 'get_encoding', 'get_streamable_parser_for_assistant']
