# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from ..block_manager import BaseBlockManager
    from ..block_trie import BlockTrie
    from ..kv_load_coordinator import KVLoadCoordinator
    from ..state_manager import StateManager

logger = get_logger('lmdeploy')


def build_eviction_helper(
    eviction_type: str,
    *,
    block_manager: BaseBlockManager,
    block_trie: BlockTrie,
    state_manager: StateManager,
    load_coordinator: KVLoadCoordinator,
    is_ssm: bool,
):
    """Build eviction helper."""
    if eviction_type == 'copy':
        logger.warning('`copy` eviction has been deprecated, '
                       'use `recompute` instead.')
        eviction_type = 'recompute'
    if eviction_type == 'recompute':
        from .recompute_eviction_helper import RecomputeEvictionHelper
        return RecomputeEvictionHelper(
            block_manager=block_manager,
            block_trie=block_trie,
            state_manager=state_manager,
            load_coordinator=load_coordinator,
            is_ssm=is_ssm,
        )
    else:
        raise TypeError(f'Unknown eviction type: {eviction_type}')
