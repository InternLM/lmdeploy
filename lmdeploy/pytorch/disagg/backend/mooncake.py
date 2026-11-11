# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from lmdeploy.profiling import PROFILE_PHASES, trace_phase
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import MigrationBackend, MooncakeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment, PDTransferResult
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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

    def __init__(self, init_request: DistServeInitRequest, shared: 'MooncakeMigrationManagement | None' = None):
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

        self.engine = shared.engine if shared is not None else TransferEngine()
        self.hostname = get_local_ip_by_remote()

        # Get all RDMA information once during initialization
        self.ibv_devices = get_rdma_nics()

        self.local_kv_table: dict[str, dict] = shared.local_kv_table if shared is not None else {}
        self.remote_kv_table: dict[str, dict] = {}
        self.remote_url: str = ''  # Store remote URL for this connection

        # Initialize the p2p connection
        if shared is None:
            self._initialize_p2p(init_request)

        self.port: int = self.engine.get_rpc_port()
        self.executor = shared.executor if shared is not None else ThreadPoolExecutor(max_workers=2,
                                                                                     thread_name_prefix='pd-transfer')
        self.slots = shared.slots if shared is not None else asyncio.Semaphore(2)
        self.pending_tickets = shared.pending_tickets if shared is not None else []
        self.has_drain_ticket = hasattr(self.engine, 'scatter_transfer_sync_read_with_ticket')

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

        mr_key = str(register_mr_request.mr_key)
        info = dict(addr=buffer_addr, length=buffer_length, offset=register_mr_request.offset)
        if mr_key in self.local_kv_table:
            if self.local_kv_table[mr_key] != info:
                raise ValueError('Cannot replace a registered PD allocation before drain.')
            return
        result = self.engine.register_memory(buffer_addr, buffer_length)
        if result != 0:
            raise RuntimeError(f'Failed to register memory region: {result}')
        self.local_kv_table[mr_key] = info

        logger.info(f'Registered memory region with mr_key {mr_key}, '
                    f'addr: {buffer_addr}, length: {buffer_length} for remote_engine_id {self.remote_engine_id}')

    @property
    def endpoint_info(self) -> dict:
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
        """Keep the bounded native job alive even if its Python caller
        cancels."""
        async with self.slots:
            future = asyncio.get_running_loop().run_in_executor(self.executor, self._migrate, assignment)
            while True:
                try:
                    return await asyncio.shield(future)
                except asyncio.CancelledError:
                    # Logical cancellation belongs to paging. Return the physical
                    # outcome before releasing the worker slot, even on shutdown.
                    continue

    def _migrate(self, assignment: MigrationAssignment) -> PDTransferResult:
        begin = perf_counter()
        if not self.remote_url:
            raise RuntimeError(f'No connection established to remote engine {self.remote_engine_id}')
        local_addresses, remote_addresses, lengths = [], [], []
        fragments = {}
        for task in assignment.batch:
            key = str(task.mr_key)
            local, remote = self.local_kv_table[key], self.remote_kv_table[key]
            if (task.source_offset < 0 or task.target_offset < 0 or task.length <= 0
                    or task.source_offset + task.length > local['length']
                    or task.target_offset + task.length > remote['length']):
                raise ValueError('PD transfer exceeds its registered allocation.')
            local_addresses.append(local['addr'] + task.source_offset)
            remote_addresses.append(remote['addr'] + task.target_offset)
            lengths.append(task.length)
            fragments.setdefault(key, []).append(task)
        result = PDTransferResult(transfer_id=assignment.transfer_id, status='completed')
        if not lengths:
            return result
        prepared = perf_counter()
        # This is the native exception boundary. Once submission may have
        # happened, an exception does not prove that DMA has stopped.
        try:
            if self.has_drain_ticket:
                keys = list(fragments)
                ticket = self.engine.scatter_transfer_sync_read_with_ticket(
                    self.remote_url, [self.local_kv_table[k]['addr'] for k in keys],
                    [self.local_kv_table[k]['length'] for k in keys],
                    [self.remote_kv_table[k]['addr'] for k in keys],
                    [self.remote_kv_table[k]['length'] for k in keys],
                    [[t.source_offset for t in fragments[k]] for k in keys],
                    [[t.target_offset for t in fragments[k]] for k in keys],
                    [[t.length for t in fragments[k]] for k in keys])
                if not ticket.drained:
                    self.pending_tickets.append(ticket)
                    result.status = 'completion_unknown'
                elif str(ticket.status).split('.')[-1] != 'COMPLETED':
                    result.status = 'failed_drained'
            else:
                code = self.engine.batch_transfer_sync_read(self.remote_url, local_addresses, remote_addresses,
                                                           lengths)
                if code != 0:
                    result.status = 'completion_unknown'
                    result.error = f'Mooncake batch read returned {code}; drain is unknown.'
        except Exception as error:
            result.status = 'completion_unknown'
            result.error = str(error)
        if PROFILE_PHASES:
            trace_phase('pd_native', transfer_id=assignment.transfer_id, rank=self.rank, segments=len(lengths),
                        bytes=sum(lengths), address_ms=(prepared - begin) * 1000,
                        native_ms=(perf_counter() - prepared) * 1000, status=result.status)
        return result


@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):
    """Mooncake backend that manages multiple migration connections."""

    def __init__(self):
        self.links: dict[str, MooncakeMigrationManagement] = {}
        self.shared: MooncakeMigrationManagement | None = None
        self.quarantined = False

    def p2p_initialize(self, init_request: DistServeInitRequest):
        link = MooncakeMigrationManagement(init_request, self.shared)
        self.links[init_request.remote_engine_id] = link
        if self.shared is None:
            self.shared = link

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return self.links[remote_engine_id].endpoint_info

    def p2p_connect(self, remote_engine_id: str, connect_request: DistServeKVTransferEndpointInfo):
        self.links[remote_engine_id].connect(connect_request)

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        if self.quarantined:
            raise RuntimeError('PD transport requires restart after an undrained transfer.')
        result = await self.links[assignment.remote_engine_id].p2p_migrate(assignment, async_op=async_op)
        if result.status == 'completion_unknown':
            self.quarantined = True
        return result

    def close(self):
        if self.quarantined:
            raise RuntimeError('Cannot unregister PD allocations while native completion is unknown.')
        if self.shared is None:
            return
        self.shared.executor.shutdown(wait=True)
        for info in self.shared.local_kv_table.values():
            result = self.shared.engine.unregister_memory(info['addr'])
            if result != 0:
                raise RuntimeError(f'Failed to unregister PD memory: {result}')
        self.shared.local_kv_table.clear()
        self.links.clear()
        self.shared = None

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
