# Copyright (c) OpenMMLab. All rights reserved.

from ..config import BackendConfig, SpecDecodeConfig
from ..distributed import DistContext


def build_spec_agent(specdecode_config: SpecDecodeConfig,
                     backend_config: BackendConfig,
                     dist_ctx: DistContext,
                     inputs_strategy,
                     agent_strategy,
                     device: str = 'cuda'):
    """Build spec agent."""
    enable = dist_ctx.rank % dist_ctx.dist_config.attn_tp == 0 and specdecode_config is not None
    if enable:
        from .spec_agent import SpecModelAgent
        return SpecModelAgent(specdecode_config, backend_config, inputs_strategy, agent_strategy, device=device)
    else:
        from .base import BaseSpecModelAgent
        return BaseSpecModelAgent()


__all__ = ['build_spec_agent']
