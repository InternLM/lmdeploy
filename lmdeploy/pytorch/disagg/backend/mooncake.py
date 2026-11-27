# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from mooncake.engine import ScatterTransferTicket

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


class MooncakeTransferResources:
    """Own local PD transport resources for one worker.

    The backend creates and closes these resources; peer connections borrow them for the lifetime of the registered
    cache allocations.
    """

    def __init__(self, rank: int):
        try:
            num_threads = int(os.getenv('LMDEPLOY_PD_TRANSFER_THREADS', '4'))
        except ValueError as e:
            raise ValueError('LMDEPLOY_PD_TRANSFER_THREADS must be a positive integer.') from e
        if num_threads <= 0:
            raise ValueError('LMDEPLOY_PD_TRANSFER_THREADS must be a positive integer.')

        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError('Please install mooncake by following the instructions at '
                              'https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md '
                              'to run LMDeploy with MooncakeBackend.') from e

        self.engine = TransferEngine()
        self.hostname = get_local_ip_by_remote()
        ibv_devices = get_rdma_nics()
        # TODO: Support more types of metadata_server
        # e.g. "etcd://192.168.0.137:2379"
        metadata_server = 'P2PHANDSHAKE'
        protocol = 'rdma'
        if not ibv_devices:
            raise RuntimeError('No RDMA devices available')
        device_name = ibv_devices[rank % len(ibv_devices)]
        result = self.engine.initialize(self.hostname, metadata_server, protocol, device_name)
        if result != 0:
            raise RuntimeError(f'Failed to initialize Mooncake engine: {result}')

        self.port: int = self.engine.get_rpc_port()
        self.local_kv_table: dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='pd-transfer')
        self.slots = asyncio.Semaphore(num_threads)
        self.pending_jobs = set()
        self.pending_tickets = set()
        self.has_drain_ticket = hasattr(self.engine, 'scatter_transfer_sync_read_with_ticket')
        self.has_unknown_completion = False

        logger.info(f'Mooncake engine initialized for rank {rank} '
                    f'with hostname {self.hostname}, RPC port: {self.port}, PD transfer threads: {num_threads}')

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        """Register a local allocation once for all peer connections."""
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
                    f'addr: {buffer_addr}, length: {buffer_length}')

    @property
    def endpoint_info(self) -> dict:
        """Describe the local endpoint and its registered allocations."""
        mr_info = {}
        for mr_key, buffer_info in self.local_kv_table.items():
            mr_info[mr_key] = {
                'addr': buffer_info['addr'],
                'length': buffer_info['length'],
                'offset': buffer_info['offset']
            }

        endpoint_info = {'mr_info': mr_info, 'session_id': f'{self.hostname}:{self.port}'}

        logger.info('Generated local Mooncake endpoint info: '
                    f"session_id={endpoint_info['session_id']}, "
                    f'mr_count={len(mr_info)}')

        return endpoint_info

    def close(self):
        """Unregister allocations only after all peer transfers have
        drained."""
        if self.has_unknown_completion or self.pending_jobs or self.pending_tickets:
            raise RuntimeError('Cannot unregister PD allocations while native completion is unknown.')
        self.executor.shutdown(wait=True)
        for info in self.local_kv_table.values():
            result = self.engine.unregister_memory(info['addr'])
            if result != 0:
                raise RuntimeError(f'Failed to unregister PD memory: {result}')
        self.local_kv_table.clear()


class MooncakeMigrationManagement:
    """Manage one peer connection using the worker's local resources."""

    def __init__(self, init_request: DistServeInitRequest, resources: MooncakeTransferResources):
        self.resources = resources
        self.remote_engine_config: MooncakeEngineConfig = init_request.remote_engine_config
        self.remote_engine_id = init_request.remote_engine_id
        self.remote_kv_table: dict[str, dict] = {}
        self.remote_url: str = ''

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
        resources = self.resources
        try:
            timeout = max(0, assignment.deadline - time()) if assignment.deadline else None
            await asyncio.wait_for(resources.slots.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return PDTransferResult(transfer_id=assignment.transfer_id, status='failed_drained',
                                    error='PD submission deadline expired waiting for a native slot.')
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(resources.executor, self._migrate, assignment)
            resources.pending_jobs.add(future)
            try:
                result, ticket = await self._wait_native(future)
            finally:
                resources.pending_jobs.remove(future)
        finally:
            resources.slots.release()
        if ticket is None:
            return result
        if result.status != 'completion_unknown':
            resources.pending_tickets.remove(ticket)
            return result

        # An unknown ticket owns native work; dropping it can block with the GIL
        # held. Poll without occupying a submission slot or a native thread while
        # idle, so an unhealthy peer does not prevent healthy transfers.
        try:
            while not ticket.drained:
                try:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    pass
                await self._wait_native(loop.run_in_executor(resources.executor, ticket.drain, 0))
        except Exception as error:
            result.error = f'Mooncake scatter drain failed: {error}'
            return result
        result.status = ticket.status.name.lower()
        result.error = '' if result.status == 'completed' else 'Mooncake scatter read failed after draining.'
        resources.pending_tickets.remove(ticket)
        return result

    @staticmethod
    async def _wait_native(future):
        # Paging owns logical cancellation and its 300-second deadline. Shield
        # each native Future directly: a child Task would also be cancelled by
        # worker shutdown, losing the physical outcome and releasing its slot.
        while True:
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                if future.cancelled():
                    raise

    def _migrate(self, assignment: MigrationAssignment) -> tuple[PDTransferResult, 'ScatterTransferTicket | None']:
        resources = self.resources
        if assignment.deadline and time() >= assignment.deadline:
            return PDTransferResult(transfer_id=assignment.transfer_id, status='failed_drained',
                                    error='PD submission deadline expired before native I/O.'), None
        if not self.remote_url:
            raise RuntimeError(f'No connection established to remote engine {self.remote_engine_id}')
        local_addresses, remote_addresses, lengths = [], [], []
        fragments = {}
        for task in assignment.batch:
            key = str(task.mr_key)
            local, remote = resources.local_kv_table[key], self.remote_kv_table[key]
            if (task.source_offset < 0 or task.target_offset < 0 or task.length <= 0
                    or task.source_offset + task.length > local['length']
                    or task.target_offset + task.length > remote['length']):
                raise ValueError('PD transfer exceeds its registered allocation.')
            if not resources.has_drain_ticket:
                local_addresses.append(local['addr'] + task.source_offset)
                remote_addresses.append(remote['addr'] + task.target_offset)
                lengths.append(task.length)
            fragments.setdefault(key, []).append(task)
        result = PDTransferResult(transfer_id=assignment.transfer_id, status='completed')
        if not fragments:
            return result, None
        if assignment.deadline and time() >= assignment.deadline:
            return PDTransferResult(transfer_id=assignment.transfer_id, status='failed_drained',
                                    error='PD submission deadline expired while preparing addresses.'), None
        ticket = None
        # This is the native exception boundary. Once submission may have
        # happened, an exception does not prove that DMA has stopped.
        try:
            if resources.has_drain_ticket:
                keys = list(fragments)
                ticket = resources.engine.scatter_transfer_sync_read_with_ticket(
                    self.remote_url, [resources.local_kv_table[k]['addr'] for k in keys],
                    [resources.local_kv_table[k]['length'] for k in keys],
                    [self.remote_kv_table[k]['addr'] for k in keys],
                    [self.remote_kv_table[k]['length'] for k in keys],
                    [[t.source_offset for t in fragments[k]] for k in keys],
                    [[t.target_offset for t in fragments[k]] for k in keys],
                    [[t.length for t in fragments[k]] for k in keys])
                resources.pending_tickets.add(ticket)
                result.status = ticket.status.name.lower()
                if result.status != 'completed':
                    result.error = f'Mooncake scatter read to {self.remote_url}: {result.status}.'
            else:
                code = resources.engine.batch_transfer_sync_read(self.remote_url, local_addresses, remote_addresses,
                                                                lengths)
                if code != 0:
                    result.status = 'completion_unknown'
                    result.error = f'Mooncake batch read returned {code}; drain is unknown.'
        except Exception as error:
            result.status = 'completion_unknown'
            result.error = str(error)
        return result, ticket


@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):
    """Own worker-local resources and the peer connections that borrow them."""

    def __init__(self):
        self.links: dict[str, MooncakeMigrationManagement] = {}
        self.resources: MooncakeTransferResources | None = None

    def p2p_initialize(self, init_request: DistServeInitRequest):
        if self.resources is None:
            self.resources = MooncakeTransferResources(rank=init_request.rank)
        self.links[init_request.remote_engine_id] = MooncakeMigrationManagement(init_request, resources=self.resources)

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.resources.register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: str, protocol: MigrationProtocol):
        return self.resources.endpoint_info

    def p2p_connect(self, remote_engine_id: str, connect_request: DistServeKVTransferEndpointInfo):
        self.links[remote_engine_id].connect(connect_request)

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        result = await self.links[assignment.remote_engine_id].p2p_migrate(assignment, async_op=async_op)
        if result.status == 'completion_unknown':
            # Block reuse follows the scheduler's bounded quarantine policy.
            # Whole-pool unregistration still requires physical completion.
            self.resources.has_unknown_completion = True
        return result

    def close(self):
        if self.resources is None:
            return
        self.resources.close()
        self.links.clear()
        self.resources = None

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
