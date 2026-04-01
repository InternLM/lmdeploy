# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine import Registry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.openai.reasoning_parser'])


@ReasoningParserManager.register_module(name=[
    'qwen-qwq', 'qwen3', 'intern-s1', 'deepseek-r1'
])
class ReasoningParser:
    """Unified reasoning parser for all ``--reasoning-parser`` options."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        self.model_tokenizer = tokenizer

    def get_reasoning_open_tag(self) -> str | None:
        return '<think>'

    def get_reasoning_close_tag(self) -> str | None:
        return '</think>'

    def starts_in_reasoning_mode(self) -> bool:
        return True
