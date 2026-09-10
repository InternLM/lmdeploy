# Copyright (c) OpenMMLab. All rights reserved.
"""Thin model-agent hooks for worker-side KV connectors."""

from lmdeploy.pytorch.kv_connector import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorOutput,
)


def start_kv_connector_step(
    connector: KVConnectorBase | None,
    metadata: KVConnectorMetadata | None,
) -> bool:
    """Bind and submit connector work before model execution."""
    if metadata is None:
        return False
    if connector is None:
        raise RuntimeError('received KV connector metadata without a worker connector')
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv()
    return True


def start_kv_connector_save(
    connector: KVConnectorBase | None,
    connector_step: bool,
) -> None:
    """Submit saves immediately after model work has been queued."""
    if not connector_step:
        return
    assert connector is not None
    connector.start_save_kv()


def finish_kv_connector_step(
    connector: KVConnectorBase | None,
    connector_step: bool,
) -> KVConnectorOutput | None:
    """Collect rank-local progress after model execution has been launched."""
    if not connector_step:
        return None
    assert connector is not None
    output = connector.get_finished()
    connector.clear_connector_metadata()
    return output
