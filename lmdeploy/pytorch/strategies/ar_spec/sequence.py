# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (
    InputEmbeddings,
    MessageStatus,
    MultiModalInputs,
    SamplingParam,
    SchedulerSession,
    UpdateTokenMode,
    _to_ndarray,
)
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..ar.sequence import ARSequenceStrategy, SchedulerSequenceDefault

SeqList = list['SchedulerSequenceARSpec']


@dataclass
class SchedulerSequenceARSpec(SchedulerSequenceDefault):

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        self._num_new_valid: int = 0
        self._num_valid_ids: int = len(self.history_cache)
        self._strategy: ARSpecSequenceStrategy = self._seq_meta.strategy

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def routed_experts(self) -> np.ndarray:
        if (not self.return_routed_experts) or self.all_routed_experts is None:
            return None

        end = max(0, self.num_valid_ids - 1)
        if 0 < end <= len(self.all_routed_experts):
            return self.all_routed_experts.get_real()[:end]
        else:
            return None

    @property
    def generated_ids(self) -> np.ndarray:
        end = self.num_valid_ids
        start = end - self.num_new_tokens
        return self.history_cache[start:end]

    def _update_token_ids_inputs(self, token_ids: np.ndarray):
        """Append tokens."""
        num_tokens = len(token_ids)
        self.output_start_pos = self.num_valid_ids + num_tokens
        self._num_valid_ids = self._num_valid_ids + num_tokens
        self._num_token_ids = num_tokens
        self.num_new_tokens = 0
        self.history_cache.append(token_ids)

    def _update_token_ids_prefill(self, token_ids: np.ndarray, draft_token_ids: np.ndarray,
                                  stop_pos: int = -1, routed_experts: np.ndarray = None):
        """Update token ids for prefill."""
        num_valid = len(token_ids)
        self.history_cache.append(token_ids)
        self.append_routed_experts(routed_experts)
        self._num_history_ids += self._num_token_ids
        self.num_new_tokens += num_valid
        self._num_valid_ids = self.num_history_ids + num_valid
        self._num_token_ids = num_valid
        if stop_pos == -1:
            # not stopping, add drafted tokens
            self._num_token_ids += len(draft_token_ids)
            self.history_cache.append(draft_token_ids)

    def _update_token_ids_decode(self, token_ids: np.ndarray, draft_token_ids: np.ndarray,
                                 stop_pos: int = -1, routed_experts: np.ndarray = None):
        """Update token ids for decode."""
        # back to last valid position
        self.history_cache.resize(self.num_valid_ids)

        valid_ids = token_ids[token_ids > -1]
        if stop_pos > -1:
            valid_ids = valid_ids[:stop_pos+1]

        num_valid = len(valid_ids)
        self.num_new_tokens += num_valid
        self._num_valid_ids += num_valid
        self._num_history_ids = self.num_valid_ids - 1
        # append the last accepted tokens
        self.history_cache.append(valid_ids)
        # append valid experts
        if routed_experts is not None:
            routed_experts = routed_experts[:num_valid]
            self.append_routed_experts(routed_experts)

        if stop_pos > -1:
            self._num_token_ids = 1
        else:
            # add new draft tokens if not stopped
            self.history_cache.append(draft_token_ids)
            self._num_token_ids = 1 + len(draft_token_ids)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: list[InputEmbeddings] = None,
                         model_meta: dict[str, Any] = None,
                         draft_token_ids: Tensor = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         routed_experts: np.ndarray = None,
                         stop_pos: int = -1,
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
            self._update_token_ids_prefill(token_ids, draft_token_ids,
                                           stop_pos=stop_pos, routed_experts=routed_experts)
        else:
            self._update_token_ids_decode(token_ids, draft_token_ids,
                                          stop_pos=stop_pos, routed_experts=routed_experts)
        if model_meta is not None:
            self.model_meta = model_meta

        self._update_mrope_pos_ids()

    def set_step(self, step: int):
        """Set step."""
        num_valid_ids = self.num_valid_ids
        # update step for vlm
        if len(self.history_embeddings) > 0:
            new_step, self._num_history_images, self._num_images = \
                self.history_embeddings.get_step(step)
            assert 0 <= new_step <= step
            step = new_step
        self._num_history_ids = step
        self._num_token_ids = num_valid_ids - step
        self.num_ignored_history = min(step, self.num_ignored_history)

        self.history_cache.resize(num_valid_ids)
        self.model_meta = None

        if self.return_routed_experts:
            # chunk long context might not have all routed experts
            if len(self.all_routed_experts) > step:
                self.all_routed_experts.resize(step)


class ARSpecSequenceStrategy(ARSequenceStrategy):

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: MigrationRequest | None = None,
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
            num_tokens = delta.seq_length.tolist()
        else:
            is_decoding = model_inputs.is_decoding
            num_tokens = model_inputs.seq_length.tolist()

        all_routed_experts = [None] * len(num_tokens)
        if batched_outputs.all_routed_experts is not None:
            all_routed_experts = batched_outputs.all_routed_experts.split(num_tokens, dim=0)
            all_routed_experts = [experts.numpy() for experts in all_routed_experts]

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        if extra_outputs is None or extra_outputs.draft_token_ids is None:
            draft_token_ids = [None] * batch_size
        else:
            draft_token_ids = extra_outputs.draft_token_ids.numpy()
        stop_pos = stop_pos.tolist()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL

        for idx, token in enumerate(next_token_ids):
            routed_experts = all_routed_experts[idx]
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            if msg.status != MessageStatus.RUNNING:
                continue
            cur_draft_tokens = draft_token_ids[idx]
            # fill token
            msg.update_token_ids(token,
                                 draft_token_ids=cur_draft_tokens,
                                 model_meta=model_meta,
                                 mode=update_mode,
                                 routed_experts=routed_experts,
                                 stop_pos=stop_pos[idx])
            if stop:
                msg.state.finish()
