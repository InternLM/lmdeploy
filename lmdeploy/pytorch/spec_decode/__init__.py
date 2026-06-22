# Copyright (c) OpenMMLab. All rights reserved.

from ..config import BackendConfig, MiscConfig, SpecDecodeConfig
from ..distributed import DistContext


def build_spec_agent(specdecode_config: SpecDecodeConfig,
                     backend_config: BackendConfig,
                     dist_ctx: DistContext,
                     inputs_strategy,
                     agent_strategy,
                     misc_config: MiscConfig,
                     device: str = 'cuda'):
    """Build spec agent."""
    enable = False
    if specdecode_config is not None:
        draft_tp = specdecode_config.dist_config.tp
        main_tp = dist_ctx.dist_config.tp
        enable = True
        if main_tp > 1 and draft_tp == 1 and dist_ctx.rank % main_tp != 0:
            enable = False
    if enable:
        from .spec_agent import SpecModelAgent
        return SpecModelAgent(specdecode_config,
                              backend_config,
                              inputs_strategy,
                              agent_strategy,
                              misc_config,
                              dist_ctx,
                              device=device)
    else:
        from .base import BaseSpecModelAgent
        return BaseSpecModelAgent(specdecode_config,
                              backend_config,
                              inputs_strategy,
                              agent_strategy,
                              misc_config,
                              dist_ctx,
                              device=device)


__all__ = ['build_spec_agent']
