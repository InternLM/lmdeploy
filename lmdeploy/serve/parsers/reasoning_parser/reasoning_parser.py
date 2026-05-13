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

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):

        vocab = tokenizer.get_vocab()
        start_token_id = vocab.get(self.get_reasoning_open_tag())
        end_token_id = vocab.get(self.get_reasoning_close_tag())
        if start_token_id is None or end_token_id is None:
            raise RuntimeError(f'{self.__class__.__name__} reasoning parser could not get '
                               'reasoning tokens from the tokenizer!')

    def get_reasoning_open_tag(self) -> str | None:
        return '<think>'

    def get_reasoning_close_tag(self) -> str | None:
        return '</think>'

    def starts_in_reasoning_mode(self) -> bool:
        return True
