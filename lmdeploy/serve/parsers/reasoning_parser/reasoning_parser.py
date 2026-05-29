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
        start_token_id = tokenizer.convert_tokens_to_ids(cls.get_reasoning_open_tag())
        end_token_id = tokenizer.convert_tokens_to_ids(cls.get_reasoning_close_tag())
        if start_token_id is None or end_token_id is None:
            raise RuntimeError(f'{cls.__name__} reasoning parser could not get '
                'reasoning tokens from the tokenizer!')

    @classmethod
    def get_reasoning_open_tag(cls) -> str:
        return '<think>'

    @classmethod
    def get_reasoning_close_tag(cls) -> str:
        return '</think>'

    def starts_in_reasoning_mode(self) -> bool:
        return True
