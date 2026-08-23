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
    connector.handle_preemptions(metadata)
    connector.start_load_kv()
    return True


def finish_kv_connector_step(
    connector: KVConnectorBase | None,
    connector_step: bool,
) -> KVConnectorOutput | None:
    """Collect rank-local progress after model execution has been launched."""
    if not connector_step:
        return None
    assert connector is not None
    finished_sending, finished_receiving = connector.get_finished(set())
    output = KVConnectorOutput(
        finished_sending=finished_sending,
        finished_receiving=finished_receiving,
        invalid_block_ids=connector.get_block_ids_with_load_errors(),
    )
    connector.clear_connector_metadata()
    return output
