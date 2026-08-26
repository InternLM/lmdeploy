# Copyright (c) OpenMMLab. All rights reserved.

from ..config import BackendConfig, MiscConfig, SpecDecodeConfig
from ..distributed import DistContext


def build_spec_agent(specdecode_config: SpecDecodeConfig,
                     backend_config: BackendConfig,
                     dist_ctx: DistContext,
                     inputs_strategy,
                     agent_strategy,
                     misc_config: MiscConfig,
                     device: str = 'cuda',
                     guided_decoding_manager=None):
    """Build a rank-local speculative decoding agent.

    Args:
        specdecode_config (SpecDecodeConfig): Speculative decoding config. If
            ``None``, speculative decoding is disabled.
        backend_config (BackendConfig): Backend config used by the draft model.
        dist_ctx (DistContext): Rank-local distributed context of the target
            model.
        inputs_strategy: Strategy for preparing model inputs.
        agent_strategy: Strategy for model-agent operations such as
            post-broadcast.
        misc_config (MiscConfig): Misc runtime config.
        device (str): Device used by the draft model.
        guided_decoding_manager: Optional guided-decoding manager. It is passed
            only to proposer-owning ``SpecModelAgent`` ranks so the draft
            proposer and target rejection sampling share one guided helper.

    Returns:
        A ``SpecModelAgent`` on ranks that own a draft proposer. Returns a
        ``BaseSpecModelAgent`` when speculative decoding is disabled, or on
        follower ranks that participate in target-side broadcasts but do not
        own a draft proposer.
    """
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
                              device=device,
                              guided_decoding_manager=guided_decoding_manager)

    from .base import BaseSpecModelAgent
    return BaseSpecModelAgent(specdecode_config,
                              backend_config,
                              inputs_strategy,
                              agent_strategy,
                              misc_config,
                              dist_ctx,
                              device=device)


__all__ = ['build_spec_agent']
