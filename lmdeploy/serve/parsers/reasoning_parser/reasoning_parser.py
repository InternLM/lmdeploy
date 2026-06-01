# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine import Registry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.parsers.reasoning_parser'])


@ReasoningParserManager.register_module(name='default')
class ReasoningParser:
    """Unified reasoning parser for all ``--reasoning-parser`` options."""

    def __init__(self, **kwargs):
        pass

    @classmethod
    def validate_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> None:
        """Validate static reasoning tags once at parser setup time."""
        vocab = tokenizer.get_vocab()
        missing_tags = [
            tag for tag in (cls.get_reasoning_open_tag(), cls.get_reasoning_close_tag())
            if tag not in vocab
        ]
        if missing_tags:
            raise RuntimeError(
                f'{cls.__name__} reasoning parser could not get reasoning tokens '
                f'from the tokenizer: {missing_tags!r}')

    @classmethod
    def get_reasoning_open_tag(cls) -> str:
        return '<think>'

    @classmethod
    def get_reasoning_close_tag(cls) -> str:
        return '</think>'

    def starts_in_reasoning_mode(self) -> bool:
        return True
