# Copyright (c) OpenMMLab. All rights reserved.
"""Factory for external KV-cache connectors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import KVConnectorBase, KVConnectorRole

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig, DistConfig


def prepare_kv_connector_config(
    cache_config: CacheConfig,
    *,
    model_path: str | None = None,
    dist_config: DistConfig | None = None,
    distributed_executor_backend: str | None = None,
) -> None:
    """Prepare runtime values before the config is copied to workers."""
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is None or not transfer_config.is_kv_transfer_instance:
        return
    if transfer_config.kv_connector == 'MooncakeStoreConnector':
        if distributed_executor_backend == 'mp':
            raise ValueError(
                'Mooncake Store does not support distributed_executor_backend="mp"; '
                'use "ray" for multi-GPU execution')

        extra_config = transfer_config.kv_connector_extra_config
        if transfer_config.is_kv_consumer:
            lookup_async = extra_config.get('lookup_async', True)
            if lookup_async is not True:
                raise ValueError('Mooncake Store requires lookup_async=true')
            extra_config['lookup_async'] = True

        cache_prefix = extra_config.get('cache_prefix', '')
        if not isinstance(cache_prefix, str):
            raise TypeError('cache_prefix must be a string')

        if 'model_name' in extra_config:
            model_name = extra_config['model_name']
            if not isinstance(model_name, str) or not model_name:
                raise ValueError('model_name must be a non-empty string')
        elif model_path is not None:
            resolved_model_path = os.path.realpath(model_path)
            default_model_name = os.path.basename(resolved_model_path.rstrip(os.sep))
            if not default_model_name:
                raise ValueError('cannot derive model_name from model_path')
            extra_config['model_name'] = default_model_name

        if transfer_config.is_kv_consumer:
            from .mooncake.store.lookup import prepare_lookup_rpc_path

            dp_rank = 0 if dist_config is None else dist_config.dp_rank
            dp_size = 1 if dist_config is None else dist_config.dp
            prepare_lookup_rpc_path(
                cache_config,
                dp_rank=dp_rank,
                dp_size=dp_size,
            )


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
