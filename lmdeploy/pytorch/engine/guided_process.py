# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger('lmdeploy')


class GuidedDecodingManager:
    processors = {}

    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: Optional[int]):
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info)
        self.vocab_size = vocab_size

    def get_processors(self, session_ctx: List[Dict[str, Any]],
                       response_formats: Tuple[Dict]) -> Dict[int, xgr.GrammarMatcher]:
        processors = {}
        for i, _format in enumerate(response_formats):
            if isinstance(_format, Dict) and _format.get('type', 'text') != 'text':
                schema_type = _format['type']
                if schema_type == 'json_schema':
                    schema = _format['json_schema']
                    if isinstance(schema, Dict):
                        for key in ['json_schema', 'schema']:
                            if key in schema:
                                schema = json.dumps(schema[key], ensure_ascii=False)

                    if not isinstance(schema, str):
                        raise ValueError(f'Cannot parse schema {schema}. The schema must be '
                                         'either a dictionary or a string that contains the'
                                         ' JSON Schema specification')
                elif schema_type == 'regex_schema':
                    schema = _format.get('regex_schema', '')
                elif schema_type == 'json_object':
                    schema = '{"type" : "object", "additionalProperties": true}'
                else:
                    raise ValueError(f'unsupported format type: {schema_type}')

                session_id = session_ctx[i]['session_id']
                seq_id = session_ctx[i]['seq_id']

                processors[i] = self.get_processor(session_id, seq_id, schema, schema_type)

        return processors

    def get_processor(self, session_id: int, seq_id: int, schema: str, type: str) -> xgr.GrammarMatcher:
        if session_id in self.processors:
            session_dict = self.processors[session_id]
            if seq_id in session_dict:
                processor = session_dict[seq_id]
                return processor

        if type == 'json_schema':
            if isinstance(schema, str):
                schema = json.loads(schema)

            assert isinstance(schema, dict)
            compiled = self.compiler.compile_json_schema(schema)
        elif type == 'regex_schema':
            compiled = self.compiler.compile_regex(schema)
        elif type == 'json_object':
            compiled = self.compiler.compile_json_schema(schema)
        else:
            assert False, f'Do not support schema type {type}'

        processor = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
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
        processor.accept_token(token)

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
