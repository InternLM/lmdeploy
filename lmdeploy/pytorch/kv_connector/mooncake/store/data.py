# Copyright (c) OpenMMLab. All rights reserved.
"""Shared data structures for the Mooncake Store connector."""

from dataclasses import dataclass

from lmdeploy.pytorch.kv_connector.base import KVConnectorMetadata


@dataclass
class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Serializable scheduler metadata for one engine step.

    Fields will be added as lookup and asynchronous transfer support is implemented. Keeping the type concrete from the
    start lets the connector validate scheduler-to-worker metadata without guessing its future shape.
    """
