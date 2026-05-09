# Copyright (c) OpenMMLab. All rights reserved.

from contextlib import contextmanager

import torch

from ..config import BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from ..distributed import DistContext
from ..engine.logits_process import SamplingInputs
from ..model_inputs import ModelInputs
from ..strategies.base.model_agent import ExtraInputs, ModelAgentStrategy
from ..strategies.base.model_inputs import ModelInputsStrategy
from .reject_sampler import RejectionSampler


def _build_draft_dist_ctx(dist_ctx: DistContext, specdecode_config: SpecDecodeConfig) -> DistContext:
    """Build draft dist context."""
    if specdecode_config is None:
        return None

    draft_dist_config = specdecode_config.dist_config
    return DistContext.build(rank=dist_ctx.rank, dist_config=draft_dist_config)


class BaseSpecModelAgent:
    """Speculative model agent."""

    def __init__(
        self,
        specdecode_config: SpecDecodeConfig,
        backend_config: BackendConfig,
        inputs_strategy: ModelInputsStrategy,
        agent_strategy: ModelAgentStrategy,
        misc_config: MiscConfig,
        dist_ctx: DistContext,
        device: str = 'cuda',
    ):
        self._enabled = specdecode_config is not None
        self.specdecode_config = specdecode_config
        self.num_spec_tokens = specdecode_config.num_speculative_tokens if specdecode_config is not None else 0
        self.backend_config = backend_config
        self.dist_ctx = dist_ctx
        self.rank = dist_ctx.rank
        self.draft_dist_ctx = _build_draft_dist_ctx(dist_ctx, specdecode_config)
        self.device = device
        self.cache_engine = None
        self.inputs_strategy = inputs_strategy
        self.agent_strategy = agent_strategy
        self.misc_config = misc_config
        self.rejection_sampler = RejectionSampler()
        self.proposer = None
        self.model_config = specdecode_config.model_config if specdecode_config is not None else None
        self.cache_config = specdecode_config.cache_config if specdecode_config is not None else None

    def is_enabled_running(self):
        """Whether spec agent is running."""
        return self.proposer is not None

    def is_enabled(self):
        return self._enabled

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        pass

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        pass

    def build_model(self, empty_init: bool, target_model=None, build_model_ctx=None):
        """Build draft model."""
        pass

    def build_graph_runner(self):
        """Build graph runner."""
        pass

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        pass

    async def async_model_forward(self,
                                model_inputs: ModelInputs,
                                extra_inputs: ExtraInputs,
                                sampling_inputs: SamplingInputs):
        """Draft model forward."""
        return extra_inputs

    def warmup(self, max_batches: int, target_model_config: ModelConfig):
        """warmup."""
        pass

    def reset_graph_runner(self):
        'reset graph runner'
        pass

    def update_main_model_outputs(self, output: dict[str, torch.Tensor],
                                  model_inputs: ModelInputs):
        """Update outputs of main model."""
        if not self.is_enabled():
            hidden_states = output.pop('hidden_states')
            return hidden_states, output

        hidden_states = output['hidden_states']

        # use original is_decoding if dp_meta is not None
        is_decoding = model_inputs.is_decoding
        if model_inputs.dp_meta is not None:
            is_decoding = model_inputs.dp_meta.is_decoding

        if not is_decoding:
            logits_indices = model_inputs.seq_length.cumsum(0) - 1
            hidden_states = hidden_states[:, logits_indices]
        if 'aux_hidden_states' in output:
            # replace with aux
            output['hidden_states'] = output.pop('aux_hidden_states')
        return hidden_states, output

    def get_model(self):
        """Get model."""
        return None

    def get_padding_batch_size(self, num_tokens: int):
        """Get padding batch size."""
        padding_batch_size = num_tokens // (self.num_spec_tokens + 1) if self.is_enabled() else num_tokens
        return padding_batch_size

    @contextmanager
    def post_broadcast(self, extra_inputs: ExtraInputs, dist_ctx, need_broadcast: bool):
        """Post broadcast."""
        enable = need_broadcast and self.is_enabled()
        if not enable:
            yield
            return

        with self.agent_strategy.post_broadcast(extra_inputs, dist_ctx) as handle:
            yield handle
