# Copyright (c) OpenMMLab. All rights reserved.
"""Shared data structures for the Mooncake Store connector."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from lmdeploy.pytorch.kv_connector.base import KVConnectorMetadata

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024
MOONCAKE_CONFIG_PATH_ENV = 'MOONCAKE_CONFIG_PATH'

MooncakeMode = Literal['embedded']


def _parse_size(value: Any) -> int:
    """Parse a byte count or a size string using binary units."""
    if isinstance(value, bool):
        raise TypeError('Mooncake size must not be a boolean')
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f'unsupported Mooncake size type: {type(value).__name__}')

    match = re.fullmatch(r'\s*(\d+(?:\.\d*)?|\.\d+)\s*(gb|mb|kb|b)?\s*', value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f'invalid Mooncake size: {value!r}')

    multipliers = {
        'gb': 1024**3,
        'mb': 1024**2,
        'kb': 1024,
        'b': 1,
    }
    number = float(match.group(1))
    unit = (match.group(2) or 'b').lower()
    return int(number * multipliers[unit])


@dataclass(frozen=True)
class MooncakeStoreConfig:
    """Validated configuration for an embedded Mooncake Store client."""

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = 'embedded'
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False

    def __post_init__(self) -> None:
        for field_name in ('metadata_server', 'master_server_address'):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{field_name} must be a non-empty string')
        if not isinstance(self.protocol, str) or self.protocol not in ('rdma', 'tcp'):
            raise ValueError("protocol must be either 'rdma' or 'tcp'")
        if not isinstance(self.device_name, str):
            raise TypeError('device_name must be a string')
        if self.mode != 'embedded':
            raise ValueError('LMDeploy Mooncake Store currently supports only embedded mode')
        if not isinstance(self.enable_offload, bool):
            raise TypeError('enable_offload must be a boolean')
        if self.enable_offload:
            raise ValueError('LMDeploy Mooncake Store does not support SSD offload yet')
        if isinstance(self.global_segment_size, bool) or not isinstance(self.global_segment_size, int):
            raise TypeError('global_segment_size must be an integer')
        if isinstance(self.local_buffer_size, bool) or not isinstance(self.local_buffer_size, int):
            raise TypeError('local_buffer_size must be an integer')
        if self.global_segment_size <= 0:
            raise ValueError('global_segment_size must be greater than 0')
        if self.local_buffer_size <= 0:
            raise ValueError('local_buffer_size must be greater than 0')

    @classmethod
    def from_file(cls, file_path: str | os.PathLike[str]) -> MooncakeStoreConfig:
        """Load and validate a Mooncake JSON configuration file."""
        with open(file_path, encoding='utf-8') as file:
            raw_config = json.load(file)
        if not isinstance(raw_config, Mapping):
            raise TypeError('Mooncake configuration must be a JSON object')

        enable_offload = raw_config.get('enable_offload', False)
        if not isinstance(enable_offload, bool):
            raise TypeError('enable_offload must be a boolean')
        enable_ssd_offload = raw_config.get('enable_ssd_offload', False)
        if not isinstance(enable_ssd_offload, bool):
            raise TypeError('enable_ssd_offload must be a boolean')
        ssd_offload_path = raw_config.get('ssd_offload_path', '')
        if not isinstance(ssd_offload_path, str):
            raise TypeError('ssd_offload_path must be a string')
        if enable_ssd_offload or ssd_offload_path:
            raise ValueError('LMDeploy Mooncake Store does not support SSD offload yet')
        return cls(
            metadata_server=raw_config.get('metadata_server', ''),
            master_server_address=raw_config.get('master_server_address', ''),
            protocol=raw_config.get('protocol', 'rdma'),
            device_name=raw_config.get('device_name', ''),
            mode=raw_config.get('mode', 'embedded'),
            global_segment_size=_parse_size(raw_config.get('global_segment_size', DEFAULT_GLOBAL_SEGMENT_SIZE)),
            local_buffer_size=_parse_size(raw_config.get('local_buffer_size', DEFAULT_LOCAL_BUFFER_SIZE)),
            enable_offload=enable_offload,
        )

    @classmethod
    def load_from_config(
        cls,
        config_path: str | os.PathLike[str] | None = None,
    ) -> MooncakeStoreConfig:
        """Load an explicit path, falling back to ``MOONCAKE_CONFIG_PATH``."""
        resolved_path = config_path or os.getenv(MOONCAKE_CONFIG_PATH_ENV)
        if not resolved_path:
            raise ValueError(
                "Mooncake config path is required: set kv_connector_extra_config['mooncake_config_path'] "
                f'or the {MOONCAKE_CONFIG_PATH_ENV} environment variable')
        return cls.from_file(resolved_path)


@dataclass(frozen=True)
class MooncakeStoreRegistration:
    """One contiguous GPU cache row registered with Mooncake Store."""

    name: str
    address: int
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Mooncake registration name must be non-empty')
        if self.address <= 0:
            raise ValueError('Mooncake registration address must be greater than 0')
        if self.size <= 0:
            raise ValueError('Mooncake registration size must be greater than 0')


@dataclass
class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Serializable scheduler metadata for one engine step.

    Fields will be added as lookup and asynchronous transfer support is implemented. Keeping the type concrete from the
    start lets the connector validate scheduler-to-worker metadata without guessing its future shape.
    """
