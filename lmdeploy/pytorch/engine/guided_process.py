# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
from typing import Any

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerBase

from lmdeploy.serve.openai.chat_completions.guided import (
    _to_xgr_structural_tag,
)

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

    @staticmethod
    def _extract_schema(response_format: dict) -> tuple[Any, str]:
        """Extract ``(schema, schema_type)`` from a response_format dict.

        ``schema`` is normalized to the form expected by ``_compile``: a JSON
        schema string for ``json_schema``/``json_object``, a regex string for
        ``regex_schema``, and the structural_tag payload dict for
        ``structural_tag``.
        """
        schema_type = response_format['type']
        if schema_type == 'json_schema':
            schema = response_format['json_schema']
            if isinstance(schema, dict):
                for key in ['json_schema', 'schema']:
                    if key in schema:
                        val = schema[key]
                        schema = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

            if not isinstance(schema, str):
                raise ValueError(f'Cannot parse schema {schema}. The schema must be '
                                 'either a dictionary or a string that contains the'
                                 ' JSON Schema specification')
        elif schema_type == 'regex_schema':
            schema = response_format.get('regex_schema', '')
        elif schema_type == 'json_object':
            schema = '{"type" : "object", "additionalProperties": true}'
        elif schema_type == 'structural_tag':
            # structural_tag payload dict; defaults to the whole response_format
            # if the 'structural_tag' key is absent.
            schema = response_format.get('structural_tag', response_format)
        else:
            raise ValueError(f'unsupported format type: {schema_type}')
        return schema, schema_type

    def _compile(self, schema: Any, schema_type: str) -> xgr.CompiledGrammar:
        """Compile an already-extracted schema into a CompiledGrammar."""
        if schema_type == 'json_schema':
            if isinstance(schema, str):
                schema = json.loads(schema)

            assert isinstance(schema, dict)
            return self.compiler.compile_json_schema(schema)
        elif schema_type == 'regex_schema':
            return self.compiler.compile_regex(schema)
        elif schema_type == 'json_object':
            return self.compiler.compile_json_schema(schema)
        elif schema_type == 'structural_tag':
            return self.compiler.compile_structural_tag(_to_xgr_structural_tag(schema))
        else:
            raise ValueError(f'Do not support schema type {schema_type}')

    def _compile_response_format(self, response_format: dict) -> xgr.CompiledGrammar:
        """Compile a full response_format dict into a CompiledGrammar.

        Single compile entrypoint covering all supported types
        (``json_schema``/``regex_schema``/``json_object``/``structural_tag``).
        Used by ``get_processor`` and exposed for unit testing.
        """
        schema, schema_type = self._extract_schema(response_format)
        return self._compile(schema, schema_type)

    def get_processors(self, session_ctx: list[dict[str, Any]],
                       response_formats: tuple[dict]) -> dict[int, xgr.GrammarMatcher]:
        processors = {}
        for i, _format in enumerate(response_formats):
            if isinstance(_format, dict) and _format.get('type', 'text') != 'text':
                schema, schema_type = self._extract_schema(_format)

                session_id = session_ctx[i]['session_id']
                seq_id = session_ctx[i]['seq_id']

                processors[i] = self.get_processor(session_id, seq_id, schema, schema_type)

        return processors

    def get_processor(self, session_id: int, seq_id: int, schema: Any, type: str) -> xgr.GrammarMatcher:
        if session_id in self.processors:
            session_dict = self.processors[session_id]
            if seq_id in session_dict:
                processor = session_dict[seq_id]
                return processor

        compiled = self._compile(schema, type)

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
