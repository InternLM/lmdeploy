# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSequence, SchedulerSession, UpdateTokenMode, _to_ndarray)
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

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
                         routed_experts: np.ndarray = None,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        token_ids = _to_ndarray(token_ids)

        num_valid = len(token_ids)
        # record cached expert ids
        self.append_routed_experts(routed_experts)

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

        if self.return_routed_experts:
            # chunk long context might not have all routed experts
            if len(self.all_routed_experts) > step:
                self.all_routed_experts.resize(step)


class ARSequenceStrategy(SequenceStrategy):

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequence':
        """Make sequence."""
        return SchedulerSequenceDefault(
            seq_id=seq_id,
            session=session,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            migration_request=migration_request,
            resp_cache=resp_cache,
            preserve_cache=preserve_cache,
        )

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, model_inputs: 'ModelInputs',
                       delta: 'ModelInputsDelta') -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)

        next_token_ids = next_token_ids.numpy()
        if model_inputs is None:
            num_tokens = delta.seq_length.tolist()
            is_decoding = delta.is_decoding
        else:
            num_tokens = model_inputs.seq_length.tolist()
            is_decoding = model_inputs.is_decoding
        all_routed_experts = [None] * len(num_tokens)
        if batched_outputs.all_routed_experts is not None:
            all_routed_experts = batched_outputs.all_routed_experts.split(num_tokens, dim=0)
            all_routed_experts = [experts.numpy() for experts in all_routed_experts]
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for token, msg, stop, model_meta, routed_experts in zip(next_token_ids, running, stopped, model_metas,
                                                                all_routed_experts):
            if msg.status != MessageStatus.RUNNING:
                continue

            # fill token
            msg.update_token_ids(token, model_meta=model_meta, mode=update_mode, routed_experts=routed_experts)
            if stop:
                msg.state.finish()
