# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import logging
from typing import Optional

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger('lmdeploy')


class BaseLogitsProcessor:
    """Base logits processor that uses xgrammar matcher for guided decoding."""

    def __init__(self, compiled_grammar: xgr.CompiledGrammar, tokenizer_info: xgr.TokenizerInfo):
        self.matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

    def fill_bitmap(self, guided_bitmask: torch.Tensor, index: int) -> None:
        """Fill the bitmask for the next token prediction at given index."""
        self.matcher.fill_next_token_bitmask(guided_bitmask, index)

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


_guided_processors = {}


def _get_guided_logits_processor(session_id: int,
                                 seq_id: int,
                                 guide: str,
                                 tokenizer: PreTrainedTokenizerBase,
                                 type: str,
                                 vocab_size_padded: Optional[int] = None):
    if session_id in _guided_processors:
        session_dict = _guided_processors[session_id]
        if seq_id in session_dict:
            processor = session_dict[seq_id]
            return processor

    if type == 'json_schema':
        processor = JSONLogitsProcessor(guide, tokenizer, vocab_size_padded)
    elif type == 'regex_schema':
        processor = RegexLogitsProcessor(guide, tokenizer, vocab_size_padded)
    else:
        assert False, f'Do not support schema type {type}'

    _guided_processors.setdefault(session_id, {})[seq_id] = processor
    return processor


def _remove_guided_logtis_processor(session_id: int):
    if session_id in _guided_processors:
        del _guided_processors[session_id]


def _allocate_batched_bitmap(batch_size: int, vocab_size: int):
    return xgr.allocate_token_bitmask(batch_size, vocab_size)


def _apply_batched_bitmap(logits: torch.Tensor, guided_bitmask: torch.Tensor) -> None:
    device = logits.device
    dtype = logits.dtype

    if device.type in {'cpu', 'cuda'}:
        xgr.apply_token_bitmask_inplace(logits, guided_bitmask.to(device))
    else:
        cpu_logits = logits.cpu().float()
        cpu_mask = guided_bitmask.cpu()
        xgr.apply_token_bitmask_inplace(cpu_logits, cpu_mask)
        logits.copy_(cpu_logits.to(device, dtype))
