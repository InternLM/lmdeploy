# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
import socket
import subprocess
from typing import Dict

from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import MigrationBackend, MooncakeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeInitRequest, DistServeKVTransferEndpointInfo,
                                                   MigrationProtocol)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

LMDEPLOY_USE_ASYNC_MIGRATION = os.environ.get('LMDEPLOY_USE_ASYNC_MIGRATION', None)


def get_rdma_nics():
    """Get all available RDMA network interface cards on the current machine.

    Returns:
        list: List of RDMA NICs, e.g. ['erdma_0', 'erdma_1']
    """
    rdma_nics = []

    try:
        result = subprocess.run(['ibv_devices'], stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Parse ibv_devices output
            # Sample output:
            # device                 node GUID
            # ------              ----------------
            lines = result.stdout.strip().split('\n')
            for line in lines[2:]:  # Skip header lines
                if line.strip():
                    device_name = line.split()[0].strip()
                    rdma_nics.append(device_name)
    except Exception as e:
        logger.error(f'Error executing ibv_devices command: {e}')

    return rdma_nics


def get_local_ip_by_remote() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(('2001:4860:4860::8888', 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        raise ValueError('Can not get local ip')


class MooncakeMigrationManagement:
    """Manages migration for a single connection in Mooncake backend."""

    def __init__(self, init_request: DistServeInitRequest):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError('Please install mooncake by following the instructions at '
                              'https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md '
                              'to run LMDeploy with MooncakeBackend.') from e

        self.rank = init_request.rank
        self.local_engine_config: MooncakeEngineConfig = init_request.local_engine_config
        self.remote_engine_config: MooncakeEngineConfig = init_request.remote_engine_config
        self.local_engine_id = init_request.local_engine_id
        self.remote_engine_id = init_request.remote_engine_id

        self.engine = TransferEngine()
        self.hostname = get_local_ip_by_remote()

        # Get all RDMA information once during initialization
        self.ibv_devices = get_rdma_nics()

        self.local_kv_table: Dict[str, Dict] = {}
        self.remote_kv_table: Dict[str, Dict] = {}
        self.remote_url: str = ''  # Store remote URL for this connection

        # Initialize the p2p connection
        self._initialize_p2p(init_request)

        self.port: int = self.engine.get_rpc_port()

    def _initialize_p2p(self, init_request: DistServeInitRequest):
        """Initialize p2p connection for this specific link."""
        # TODO: Support more types of metadata_server
        # e.g. "etcd://192.168.0.137:2379"
        metadata_server = 'P2PHANDSHAKE'

        # Default protocol (Currently only RDMA is supported)
        protocol = 'rdma'

        # Get the device name from request
        if not self.ibv_devices:
            raise RuntimeError('No RDMA devices available')

        device_name = self.ibv_devices[self.rank % len(self.ibv_devices)]

        # Initialize the engine
        result = self.engine.initialize(self.hostname, metadata_server, protocol, device_name)
        if result != 0:
            raise RuntimeError(f'Failed to initialize Mooncake engine: {result}')

        logger.info(f'Mooncake engine initialized for remote_engine_id {self.remote_engine_id} '
                    f'with hostname {self.hostname}, RPC port: {self.engine.get_rpc_port()}')

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        """Register memory region for this connection."""
        # Transmit buffer address to int
        buffer_addr = register_mr_request.addr
        buffer_length = register_mr_request.length

        # Register memory region with the engine
        result = self.engine.register_memory(buffer_addr, buffer_length)
        if result != 0:
            raise RuntimeError(f'Failed to register memory region: {result}')

        mr_key = str(register_mr_request.mr_key)
        self.local_kv_table[mr_key] = {
            'addr': buffer_addr,
            'length': buffer_length,
            'offset': register_mr_request.offset
        }

        logger.info(f'Registered memory region with mr_key {mr_key}, '
                    f'addr: {buffer_addr}, length: {buffer_length} for remote_engine_id {self.remote_engine_id}')

    @property
    def endpoint_info(self) -> Dict:
        """Get endpoint information for this connection."""

        mr_info = {}
        for mr_key, buffer_info in self.local_kv_table.items():
            mr_info[mr_key] = {
                'addr': buffer_info['addr'],
                'length': buffer_info['length'],
                'offset': buffer_info['offset']
            }

        endpoint_info = {'mr_info': mr_info, 'session_id': f'{self.hostname}:{self.port}'}

        logger.info(f'Generated endpoint info for remote engine {self.remote_engine_id}: '
                    f"session_id={endpoint_info['session_id']}, "
                    f'mr_count={len(mr_info)}')

        return endpoint_info

    def connect(self, connect_request: DistServeKVTransferEndpointInfo):
        """Connect to the remote engine."""
        remote_endpoint_info = json.loads(connect_request.endpoint_info)

        self.remote_url = remote_endpoint_info['session_id']
        self.remote_kv_table = remote_endpoint_info['mr_info']

        logger.info(f'Received remote buffer info: {len(self.remote_kv_table)} regions')
        for mr_key, buffer_info in self.remote_kv_table.items():
            logger.debug(f"Remote buffer mr_key {mr_key}: addr=0x{buffer_info['addr']:x}, "
                         f"length={buffer_info['length']}")

        logger.info(f'Connecting to remote engine {self.remote_engine_id} at {self.remote_url}')

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        """Migrate data to the remote engine."""
        if not LMDEPLOY_USE_ASYNC_MIGRATION:
            # For synchronous migration, call the method directly
            self._migrate(assignment)
        else:
            # For asynchronous migration, use an async method
            import asyncio
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            await loop.run_in_executor(None, self._migrate, assignment)

            result = await future
            if result != 0:
                raise RuntimeError(f'Failed to perform async transfer: {result}')

    def _migrate(self, assignment: MigrationAssignment):
        """Migrate data to the remote engine synchronously."""
        if not self.remote_url:
            raise RuntimeError(f'No connection established to remote engine {self.remote_engine_id}')

        for i, task in enumerate(assignment.batch):
            mr_key = str(task.mr_key)

            if mr_key not in self.local_kv_table:
                raise RuntimeError(f'Memory region with mr_key {mr_key} not registered locally')

            if mr_key not in self.remote_kv_table:
                raise RuntimeError(f'Remote memory region with mr_key {mr_key} not registered')

            # Get local buffer information
            local_buffer_info = self.local_kv_table[mr_key]
            local_addr = local_buffer_info['addr'] + task.source_offset

            # Get remote buffer information
            remote_buffer_info = self.remote_kv_table[mr_key]
            remote_addr = remote_buffer_info['addr'] + task.target_offset

            logger.debug(f'Task {i}: Migrating {task.length} bytes')
            logger.debug(f'  Local Engine: {self.local_engine_id}')
            logger.debug(f'  Remote Engine: {assignment.remote_engine_id}')
            logger.debug(f'  MR Key: {mr_key}')
            logger.debug(f"  Local:  0x{local_buffer_info['addr']:x} + {task.source_offset} = 0x{local_addr:x}")
            logger.debug(f"  Remote: 0x{remote_buffer_info['addr']:x} + {task.target_offset} = 0x{remote_addr:x}")
            logger.debug(f'  Session: {self.remote_url}')

            result = self.engine.transfer_sync_read(
                self.remote_url,
                local_addr,
                remote_addr,
                task.length,
            )
            if result != 0:
                raise RuntimeError(f'Failed to perform sync transfer: {result}')


@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):
    """Mooncake backend that manages multiple migration connections."""

    def __init__(self):
        self.links: Dict[int, MooncakeMigrationManagement] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        self.links[init_request.remote_engine_id] = MooncakeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return self.links[remote_engine_id].endpoint_info

    def p2p_connect(self, remote_engine_id: str, connect_request: DistServeKVTransferEndpointInfo):
        self.links[remote_engine_id].connect(connect_request)

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        await self.links[assignment.remote_engine_id].p2p_migrate(assignment, async_op=async_op)

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
