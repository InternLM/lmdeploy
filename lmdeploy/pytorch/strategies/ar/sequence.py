# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch import Tensor

from lmdeploy.pytorch.disagg.conn.protocol import EncoderResult, MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSequence, SchedulerSession, UpdateTokenMode, _to_ndarray)

from ..base.sequence import SequenceStrategy

SeqList = List[SchedulerSequence]


@dataclass
class SchedulerSequenceDefault(SchedulerSequence):

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        token_ids = _to_ndarray(token_ids)

        num_valid = len(token_ids)

        if mode == UpdateTokenMode.INPUTS:
            self.arrive_time = time.perf_counter()
            self.output_start_pos = self.num_all_ids + len(token_ids)
            self._num_token_ids += num_valid
            self.num_new_tokens = 0
        else:
            self._num_history_ids += self._num_token_ids
            num_token_ids = num_valid
            self._num_token_ids = num_token_ids
            self.num_new_tokens += num_token_ids

        self.history_cache.append(token_ids)

        if model_meta is not None:
            self.model_meta = model_meta

    def set_step(self, step: int):
        """Set step."""
        num_all_ids = self.num_all_ids
        # update step for vlm
        if len(self.history_embeddings) > 0:
            new_step, self._num_history_images, self._num_images = \
                self.history_embeddings.get_step(step)
            assert 0 <= new_step <= step
            step = new_step
        self._num_history_ids = step
        self._num_token_ids = num_all_ids - step
        self.num_ignored_history = min(step, self.num_ignored_history)

        self.model_meta = None

        # cross
        if self.history_multimodals is not None:
            self._num_history_cross = self.history_multimodals.get_encoder_len(0, self.num_history_ids)
            self._num_cross = self.history_multimodals.get_encoder_len(self._num_history_ids, num_all_ids)


class ARSequenceStrategy(SequenceStrategy):

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      encoder_result: Optional[EncoderResult] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequence':
        """Make sequence."""
        return SchedulerSequenceDefault(seq_id=seq_id,
                                        session=session,
                                        sampling_param=sampling_param,
                                        adapter_name=adapter_name,
                                        migration_request=migration_request,
                                        resp_cache=resp_cache,
                                        preserve_cache=preserve_cache)

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, is_decoding: bool) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)

        next_token_ids = next_token_ids.numpy()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for token, msg, stop, model_meta in zip(next_token_ids, running, stopped, model_metas):
            if msg.status != MessageStatus.LOCKED:
                continue

            # fill token
            msg.update_token_ids(token, model_meta=model_meta, mode=update_mode)
            if stop:
                msg.status = MessageStatus.TO_BE_MIGRATED if msg.preserve_cache else MessageStatus.STOPPED
