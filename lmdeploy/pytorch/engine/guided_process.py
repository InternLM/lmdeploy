# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Any

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerBase

from lmdeploy._guided_decoding import compile_response_format

logger = logging.getLogger('lmdeploy')


class GuidedDecodingManager:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int | None):
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size

        # vocab_size must include all token IDs the model can produce (EOS, special tokens).
        # Some models have vocab_size < len(tokenizer), causing EOS to be out of bitmask range.
        tokenizer_vocab_len = len(tokenizer)
        if tokenizer_vocab_len > vocab_size:
            logger.info(f'GuidedDecodingManager: expanding vocab_size from {vocab_size} '
                        f'to {tokenizer_vocab_len}')
            vocab_size = tokenizer_vocab_len

        # XGrammar will automatically detect stop tokens from the tokenizer
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info)
        self.vocab_size = vocab_size
        self.processors: dict[int, dict[int, xgr.GrammarMatcher]] = {}

    def get_processors(self, session_ctx: list[dict[str, Any]],
                       response_formats: tuple[dict]) -> dict[int, xgr.GrammarMatcher]:
        processors = {}
        for i, _format in enumerate(response_formats):
            if isinstance(_format, dict) and _format.get('type', 'text') != 'text':
                session_id = session_ctx[i]['session_id']
                seq_id = session_ctx[i]['seq_id']
                processors[i] = self.get_processor(session_id, seq_id, _format)

        return processors

    def get_processor(self, session_id: int, seq_id: int,
                      response_format: dict[str, Any]) -> xgr.GrammarMatcher:
        if session_id in self.processors:
            session_dict = self.processors[session_id]
            if seq_id in session_dict:
                processor = session_dict[seq_id]
                return processor

        compiled = compile_response_format(self.compiler, response_format)

        processor = xgr.GrammarMatcher(compiled)
        self.processors.setdefault(session_id, {})[seq_id] = processor
        logger.info(f'create guided processor for session_id={session_id}, seq_id={seq_id}, and '
                    f'total_processors={len(self.processors)}')
        return processor

    def remove_processor(self, session_id: int):
        if session_id in self.processors:
            del self.processors[session_id]
            logger.info(
                f'delete guided processor for session_id={session_id}, and total_processors={len(self.processors)}')

    def allocate_batched_bitmap(self, batch_size: int) -> torch.Tensor:
        return xgr.allocate_token_bitmask(batch_size, self.vocab_size)

    def fill_bitmap(self, processor: xgr.GrammarMatcher, guided_bitmask: torch.Tensor, index: int) -> None:
        processor.fill_next_token_bitmask(guided_bitmask, index)

    def accept_token(self, processor: xgr.GrammarMatcher, token: int) -> None:
        processor.accept_token(token, debug_print=False)

    def is_terminated(self, processor: xgr.GrammarMatcher) -> bool:
        return processor.is_terminated()

    def apply_batched_bitmap(self, logits: torch.Tensor, guided_bitmask: torch.Tensor) -> None:
        device = logits.device
        dtype = logits.dtype

        if device.type in {'cpu', 'cuda'}:
            xgr.apply_token_bitmask_inplace(logits, guided_bitmask.to(device))
        else:
            cpu_logits = logits.cpu().float()
            cpu_mask = guided_bitmask.cpu()
            xgr.apply_token_bitmask_inplace(cpu_logits, cpu_mask)
            logits.copy_(cpu_logits.to(device, dtype))

    def clear(self) -> None:
        self.processors.clear()
        logger.info(f'clear guided processors, total_processors={len(self.processors)}')
