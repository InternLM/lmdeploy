# Copyright 2024- the Outlines developers
# This file is adapted from
# https://github.com/outlines-dev/outlines/blob/main/outlines/serve/vllm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import copy
import math
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from functools import lru_cache
from typing import DefaultDict, Dict, List, Union

import torch
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase


class RegexLogitsProcessor:

    def __init__(self, regex_string: str, tokenizer):
        """Compile the FSM that drives the regex-structured generation.

        Args:
            regex_string: A string that represents a regular expression
            tokenizer: The model's tokenizer
        """
        tokenizer = self.adapt_tokenizer(copy.deepcopy(tokenizer))
        fsm = RegexFSM(regex_string, tokenizer)
        self.fsm = fsm

    def init_state(self):
        """Initialize the FSM states."""
        self.fsm_state: DefaultDict[int, int] = defaultdict(int)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        seq_id = hash(tuple(input_ids))

        if len(input_ids) == 0:
            self.init_state()
        else:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[last_seq_id], last_token)

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state[seq_id])

        mask = torch.full((scores.shape[-1], ),
                          -math.inf,
                          device=scores.device)
        mask[allowed_tokens] = 0
        scores.add_(mask)

        return scores

    def adapt_tokenizer(self, tokenizer):
        """Adapt tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.
        """
        from outlines.integrations.utils import adapt_tokenizer
        return adapt_tokenizer(tokenizer)


class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, Dict, BaseModel], tokenizer):
        """Compile the FSM that drives the JSON-guided generation.

        Args:
            schema: A str schema that encodes the structure we want the model
                to generate
            tokenizer: The model's tokenizer
        """
        regex_string = build_regex_from_schema(schema)
        super().__init__(regex_string, tokenizer)


@lru_cache(maxsize=32)
def _get_guided_logits_processor(guide: str,
                                 tokenizer: PreTrainedTokenizerBase,
                                 type: str):
    try:
        if type == 'json_object':
            return JSONLogitsProcessor(guide, tokenizer)
        elif type == 'regex_object':
            return RegexLogitsProcessor(guide, tokenizer)
        else:
            return None
    except Exception as e:
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        logger.error(e)
        return None
