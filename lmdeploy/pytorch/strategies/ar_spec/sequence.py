# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSession, UpdateTokenMode, _to_ndarray)
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..ar.sequence import ARSequenceStrategy, SchedulerSequenceDefault

SeqList = List['SchedulerSequenceARSpec']


@dataclass
class SchedulerSequenceARSpec(SchedulerSequenceDefault):

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        self._num_spec_ids: int = 0
        self._num_new_valid: int = 0
        self._num_valid_ids: int = len(self.history_cache)
        self._strategy: ARSpecSequenceStrategy = self._seq_meta.strategy

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def num_spec_ids(self):
        return self._num_spec_ids

    @property
    def generated_ids(self) -> np.ndarray:
        end = self.num_valid_ids
        start = end - self.num_new_tokens
        return self.history_cache[start:end]

    def set_stop_pos(self, pos: int):
        val = self._num_new_valid - pos - 1
        self._num_valid_ids -= val
        self.num_new_tokens -= val
        self._num_token_ids = 1
        self._num_history_ids -= val

        self._num_spec_ids = 0
        self._num_new_valid = 0
        self.history_cache.resize(self.num_valid_ids)

    def _update_token_ids_inputs(self, token_ids: np.ndarray):
        """Append tokens."""
        num_tokens = len(token_ids)
        self.output_start_pos = self.num_valid_ids + num_tokens
        self._num_valid_ids = self.num_history_ids + num_tokens
        self._num_token_ids = num_tokens
        self.num_new_tokens = 0
        self._num_spec_ids = 0
        self._num_new_valid = 0
        self.history_cache.append(token_ids)

    def _update_token_ids_prefill(self, token_ids: np.ndarray, draft_token_ids: np.ndarray):
        """Update token ids for prefill."""
        num_valid = len(token_ids)
        self._num_spec_ids = len(draft_token_ids)
        token_ids = np.concatenate([token_ids, draft_token_ids])
        num_tokens = len(token_ids)
        self._num_history_ids += self._num_token_ids
        self._num_token_ids = num_tokens
        self.num_new_tokens += num_valid
        self._num_new_valid = num_valid
        self._num_valid_ids = self.num_history_ids + num_valid
        self.history_cache.append(token_ids)

    def _update_token_ids_decode(self, token_ids: np.ndarray, draft_token_ids: np.ndarray = None):
        """Update token ids for decode."""
        valid_ids = token_ids[token_ids > -1]
        num_valid = len(valid_ids)
        self.num_new_tokens = self.num_new_tokens + num_valid

        self._num_new_valid = num_valid
        self._num_valid_ids += num_valid
        self._num_history_ids = self.num_valid_ids - 1

        # last step has spec ids
        if self.num_spec_ids > 0:
            token_ids = valid_ids[-1:]
        else:
            token_ids = valid_ids

        num_tokens = len(token_ids)

        if draft_token_ids is not None:
            num_tokens = 1 + len(draft_token_ids)
            token_ids = np.concatenate([token_ids, draft_token_ids])
            self._num_spec_ids = len(draft_token_ids)
        else:
            self._num_spec_ids = 0

        self._num_token_ids = num_tokens
        if self.num_history_ids < len(self.history_cache):
            self.history_cache.resize(self.num_history_ids)
        self.history_cache.append(token_ids)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         draft_token_ids: Tensor = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        self.arrive_time = time.perf_counter()

        token_ids: np.ndarray = _to_ndarray(token_ids)
        if draft_token_ids is not None:
            draft_token_ids = _to_ndarray(draft_token_ids)
        if mode == UpdateTokenMode.INPUTS:
            self._update_token_ids_inputs(token_ids)
        elif mode == UpdateTokenMode.PREFILL:
            self._update_token_ids_prefill(token_ids, draft_token_ids)
        else:
            self._update_token_ids_decode(token_ids, draft_token_ids)
        if model_meta is not None:
            self.model_meta = model_meta


class ARSpecSequenceStrategy(ARSequenceStrategy):

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequenceARSpec':
        """Make sequence."""
        return SchedulerSequenceARSpec(seq_id=seq_id,
                                       session=session,
                                       sampling_param=sampling_param,
                                       adapter_name=adapter_name,
                                       migration_request=migration_request,
                                       resp_cache=resp_cache,
                                       preserve_cache=preserve_cache)

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, model_inputs: 'ModelInputs',
                       delta: 'ModelInputsDelta', **kwargs) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        extra_outputs = batched_outputs.extra_outputs
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)
        stop_pos = batched_outputs.stop_pos

        if model_inputs is None:
            is_decoding = delta.is_decoding
        else:
            is_decoding = model_inputs.is_decoding

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        if extra_outputs is None or extra_outputs.draft_token_ids is None:
            draft_token_ids = [None] * batch_size
        else:
            draft_token_ids = extra_outputs.draft_token_ids.numpy()
        stop_pos = stop_pos.tolist()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL

        for idx, token in enumerate(next_token_ids):
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            if msg.status != MessageStatus.RUNNING:
                continue
            cur_draft_tokens = draft_token_ids[idx]
            # fill token
            msg.update_token_ids(token, draft_token_ids=cur_draft_tokens, model_meta=model_meta, mode=update_mode)
            if stop:
                msg.set_stop_pos(stop_pos[idx])
                msg.state.finish()
