# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding

from lmdeploy.serve.openai.protocol import (
    DeltaMessage,
)

from . import ResponseParserManager
from .response_parser import ResponseParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


@ResponseParserManager.register_module('gpt-oss')
class GptOssResponseParser(ResponseParser):
    """Harmony stream parser for GPT-OSS (assistant role)."""

    def __init__(self, request: ChatCompletionRequest, tokenizer: PreTrainedTokenizerBase):
        self.parser = partial(StreamableParser, get_encoding(), role=Role.ASSISTANT)

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs) -> tuple[DeltaMessage | None, bool]:
        pass

    def parse_complete(self, text: str, **kwargs) -> tuple[str, list | None, str | None]:
        pass
