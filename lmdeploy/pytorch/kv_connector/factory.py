# Copyright (c) OpenMMLab. All rights reserved.
"""Factory for external KV-cache connectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import KVConnectorBase, KVConnectorRole

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig


def prepare_kv_connector_config(cache_config: CacheConfig) -> None:
    """Prepare runtime values before the config is copied to workers."""
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is None or not transfer_config.is_kv_transfer_instance:
        return
    if transfer_config.kv_connector == 'MooncakeStoreConnector':
        from .mooncake.store.worker import prepare_lookup_rpc_path

        prepare_lookup_rpc_path(cache_config)


def build_kv_connector(
    role: KVConnectorRole,
    cache_config: CacheConfig,
    *,
    global_rank: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> KVConnectorBase | None:
    """Build the configured connector without importing optional backends
    eagerly."""
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is None or not transfer_config.is_kv_transfer_instance:
        return None

    connector_name = transfer_config.kv_connector
    if connector_name == 'MooncakeStoreConnector':
        prepare_kv_connector_config(cache_config)
        from .mooncake.store.connector import MooncakeStoreConnector

        return MooncakeStoreConnector(
            role,
            cache_config,
            global_rank=global_rank,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    raise ValueError(f'Unsupported KV connector: {connector_name!r}')
