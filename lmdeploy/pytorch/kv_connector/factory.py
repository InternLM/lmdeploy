# Copyright (c) OpenMMLab. All rights reserved.
"""Factory for external KV-cache connectors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import KVConnectorBase, KVConnectorRole

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig


def prepare_kv_connector_config(cache_config: CacheConfig, model_path: str | None = None) -> None:
    """Prepare runtime values before the config is copied to workers."""
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is None or not transfer_config.is_kv_transfer_instance:
        return
    if transfer_config.kv_connector == 'MooncakeStoreConnector':
        from .mooncake.store.worker import prepare_lookup_rpc_path

        if model_path is not None:
            resolved_model_path = os.path.realpath(model_path)
            model_name = os.path.basename(resolved_model_path.rstrip(os.sep))
            extra_config = transfer_config.kv_connector_extra_config
            # ``model_namespace`` was used by the initial implementation.
            # Keep accepting an explicit value while generating vLLM-style
            # keys from the model basename by default.
            model_name = extra_config.get('model_namespace', model_name)
            extra_config.setdefault('model_name', model_name)
        prepare_lookup_rpc_path(cache_config)


def build_kv_connector(
    role: KVConnectorRole,
    cache_config: CacheConfig,
    *,
    global_rank: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    kv_head_replica_num: int = 1,
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
            kv_head_replica_num=kv_head_replica_num,
        )

    raise ValueError(f'Unsupported KV connector: {connector_name!r}')
