# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import logging
from functools import lru_cache
from typing import Optional

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger('guided_process')


class BaseLogitsProcessor:
    """Base logits processor that uses xgrammar matcher for guided decoding."""

    def __init__(self, compiled_grammar: xgr.CompiledGrammar, tokenizer_info: xgr.TokenizerInfo):
        self.matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        self.token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    def process(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply grammar constraints to logits before sampling the next
        token."""
        self.matcher.fill_next_token_bitmask(self.token_bitmask)
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        return scores

    def accept(self, token_id: int) -> bool:
        """Update matcher state after a token is generated."""
        return self.matcher.accept_token(token_id)

    def reset(self):
        """Reset matcher state for next generation."""
        self.matcher.reset()


class RegexLogitsProcessor(BaseLogitsProcessor):
    """Regex-guided logits processor using xgrammar."""

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase, vocab_size_padded: Optional[int] = None):
        tokenizer = copy.deepcopy(tokenizer)
        if vocab_size_padded is None:
            vocab_size_padded = tokenizer.vocab_size

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size_padded)

        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_regex_grammar(regex_string)

        super().__init__(compiled, tokenizer_info)


class JSONLogitsProcessor(BaseLogitsProcessor):
    """JSON-schema guided logits processor using xgrammar."""

    def __init__(self, schema: str, tokenizer: PreTrainedTokenizerBase, vocab_size_padded: Optional[int] = None):
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size_padded)
        if vocab_size_padded is None:
            vocab_size_padded = tokenizer.vocab_size

        compiler = xgr.GrammarCompiler(tokenizer_info)
        if isinstance(schema, str):
            schema = json.loads(schema)

        assert isinstance(schema, dict)
        compiled = compiler.compile_json_schema(schema)

        super().__init__(compiled, tokenizer_info)


@lru_cache(maxsize=32)
def _get_guided_logits_processor(guide: str,
                                 tokenizer: PreTrainedTokenizerBase,
                                 type: str,
                                 vocab_size_padded: Optional[int] = None):
    try:
        if type == 'json_schema':
            return JSONLogitsProcessor(guide, tokenizer, vocab_size_padded)
        elif type == 'regex_schema':
            return RegexLogitsProcessor(guide, tokenizer, vocab_size_padded)
        else:
            return None
    except Exception as e:
        logger.error(e)
        raise
