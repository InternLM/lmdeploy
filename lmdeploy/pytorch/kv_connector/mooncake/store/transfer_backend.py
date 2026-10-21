# Copyright (c) OpenMMLab. All rights reserved.
"""Device-aware transfer backends for the Mooncake Store connector.

The Mooncake Store Python API (``batch_is_exist`` / ``batch_put_from_multi_buffers``
/ ``batch_get_into_multi_buffers``) is device-agnostic: it operates on raw buffer
addresses that have been registered with the store. What differs between CUDA and
Ascend is only *how* those buffers are registered and *how* the store is set up:

* CUDA (H200): GPU HBM is exposed to the RDMA NIC through GPUDirect RDMA. Buffers
  are registered directly with the store (``store.register_buffer``) and the store
  owns its internal TransferEngine.
* Ascend (A3): NPU HBM is registered with an external ``TransferEngine``
  (``engine.register_memory``) which is then injected into the store through
  ``setup(..., engine=...)``. The store reads and writes device HBM addresses
  directly, relying on Ascend's unified-memory RDMA path. No explicit D2H/H2D copy
  is needed on this path. On this default transport a transfer that crosses the
  engine boundary (another process's segment) must export device memory
  cross-process through CANN RTS IPC + HCCL, which fails on current A3 drivers.
* Ascend fabric-mem (``ASCEND_ENABLE_USE_FABRIC_MEM=1``): the store allocates
  its segment as HIXL fabric memory -- a unified address space shared by every
  process over the A3 fabric -- so no external engine is injected, no
  client-side buffer registration happens, and cross-segment transfers (the
  load path under MLA key striping) work without CANN RTS IPC at all.

The registered addresses and the per-block scatter layout are identical across
backends, so the send/receive transfer threads (``worker_threads.py``) require no
device-specific changes. This mirrors vllm-ascend's ``ascend_store`` mooncake
backend, which transfers directly to/from NPU HBM.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from lmdeploy.utils import get_logger

from .data import MooncakeStoreConfig, MooncakeStoreRegistration

if TYPE_CHECKING:
    pass

logger = get_logger('lmdeploy')

# Fabric-mem mode is selected purely by environment, exactly like vllm-ascend's
# MooncakeBackend (`_use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"`).
ASCEND_FABRIC_MEM_ENV = 'ASCEND_ENABLE_USE_FABRIC_MEM'
# Fabric segments must be integer multiples of 1GB (vllm-ascend compatibility
# matrix: fabric_memory allocations are 1GB-granular).
FABRIC_SEGMENT_GRANULARITY = 1024 ** 3


class TransferBackend:
    """Device-specific Mooncake Store setup, registration, and stream fencing.

    Subclasses encapsulate the three points where CUDA and Ascend differ:
    ``setup_store`` (store initialization arguments), ``register_buffer`` (which
    object registers a KV-cache region) and ``make_ready_event`` (the compute-stream
    fence event). The transfer data path itself is shared by the worker threads.
    """

    device_type: str = 'cuda'

    # Name of the store segment this process registered, when the backend can
    # compute it.  Exposed for diagnostics (fabric mode names the segment with
    # the bare hostname, no ``:port``); not used to pin put placement.
    local_segment_name: str | None = None
    # True when the backend set the store up in HIXL fabric-mem mode, where
    # cross-segment transfers are native and per-segment names are not unique
    # per process.  ``register_buffer`` uses this to skip client-side buffer
    # registration on the fabric path.
    use_fabric_mem: bool = False

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        """Initialize the store and return its status code (0 on success)."""
        raise NotImplementedError

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        """Register one KV-cache region and return the status code (0 on success)."""
        raise NotImplementedError

    def make_ready_event(self) -> Any:
        """Return a fence event recorded on the compute stream."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend-owned resources (e.g. the Ascend TransferEngine)."""
        return None


class CudaTransferBackend(TransferBackend):
    """Direct GPU-HBM registration, matching the original H200 implementation.

    Buffers are registered with the store and the store owns its TransferEngine,
    exactly as before this abstraction was introduced. Kept as a distinct class so
    the CUDA path is explicit and never accidentally changes.
    """

    device_type = 'cuda'

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        return store.setup(
            local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        return store.register_buffer(registration.address, registration.size)

    def make_ready_event(self) -> Any:
        return torch.cuda.Event()


class AscendTransferBackend(TransferBackend):
    """NPU-HBM registration via an external TransferEngine (Ascend A3).

    A process-local ``TransferEngine`` is initialized once, its engine is injected
    into the store through ``setup(..., engine=...)``, and each KV-cache HBM region
    is registered with the engine via ``register_memory``. The store then transfers
    directly to and from device HBM addresses, exactly as on CUDA. This is the
    proven store-path used by vllm-ascend; it needs no host staging or D2H/H2D copy.

    When ``ASCEND_ENABLE_USE_FABRIC_MEM=1`` the backend instead follows
    vllm-ascend's fabric-mem recipe (A3+HCCS): the store allocates its segment as
    HIXL fabric memory reachable from every process, no engine is injected, no
    buffer registration happens, and segment names carry no ``:port`` suffix.
    That is what makes cross-segment transfers -- the load path under MLA key
    striping -- actually work on A3.
    """

    device_type = 'npu'

    def __init__(self) -> None:
        # Fail fast if torch_npu is unavailable: this backend is only built on
        # Ascend, where torch_npu is a required dependency.
        import torch_npu  # noqa: F401  registers torch.npu
        self._transfer_engine: Any | None = None
        self.use_fabric_mem = os.getenv(ASCEND_FABRIC_MEM_ENV, '0') == '1'

    def _get_transfer_engine(self, local_hostname: str, device_name: str) -> Any:
        if self._transfer_engine is not None:
            return self._transfer_engine
        from mooncake.engine import TransferEngine
        engine = TransferEngine()
        # The engine uses the Ascend transport protocol and a P2P handshake; the
        # store itself is configured separately with the user-supplied metadata
        # server. Mirrors vllm-ascend GlobalTE.get_transfer_engine.
        ret = engine.initialize(local_hostname, 'P2PHANDSHAKE', 'ascend',
                                device_name or '')
        if ret != 0:
            raise RuntimeError(
                f'Ascend TransferEngine.initialize failed with return code {ret}')
        self._transfer_engine = engine
        return engine

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        if self.use_fabric_mem:
            return self._setup_fabric_store(store, config, local_hostname)
        engine = self._get_transfer_engine(local_hostname, config.device_name)
        local_seg = f'{local_hostname}:{engine.get_rpc_port()}'
        # Expose the registered segment name so callers can pin Mooncake put
        # placement to this process's own segment (same-process D2H transfers).
        self.local_segment_name = local_seg
        return store.setup(
            local_hostname=local_seg,
            metadata_server=config.metadata_server,
            global_segment_size=config.global_segment_size,
            local_buffer_size=config.local_buffer_size,
            protocol=config.protocol,
            rdma_devices=config.device_name,
            master_server_addr=config.master_server_address,
            engine=engine.get_engine(),
        )

    def _setup_fabric_store(self, store: Any, config: MooncakeStoreConfig,
                            local_hostname: str) -> int:
        """Set the store up in HIXL fabric-mem mode (vllm-ascend A3+HCCS path).

        The segment is allocated by the store itself as fabric memory -- a HIXL
        unified address space over the A3 fabric that every process can reach
        directly -- so cross-segment puts and gets no longer need the CANN RTS
        IPC + HCCL device-memory export that fails on this stack. Matching
        vllm-ascend's ``MooncakeBackend`` fabric branch exactly:

        * no external engine is injected (``setup`` without ``engine=``);
        * the segment name is the bare hostname, no ``:rpc_port`` suffix;
        * ``local_buffer_size`` is unused in fabric mode and passed as 0;
        * fabric segments must be integer multiples of 1GB.
        """
        if config.global_segment_size % FABRIC_SEGMENT_GRANULARITY:
            raise ValueError(
                f'{ASCEND_FABRIC_MEM_ENV}=1 requires global_segment_size to be '
                f'an integer multiple of 1GB, got {config.global_segment_size} '
                'bytes')
        self.local_segment_name = local_hostname
        return store.setup(
            local_hostname=local_hostname,
            metadata_server=config.metadata_server,
            global_segment_size=config.global_segment_size,
            local_buffer_size=0,
            protocol=config.protocol,
            rdma_devices=config.device_name,
            master_server_addr=config.master_server_address,
        )

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        if self.use_fabric_mem:
            # HIXL exports device memory through fabric-shareable handles
            # automatically; vllm-ascend's fabric path registers nothing, and
            # addresses are consumed directly by the fabric transport.
            return 0
        engine = self._transfer_engine
        if engine is None:
            raise RuntimeError(
                'Ascend TransferEngine must be initialized by setup_store before '
                'register_buffer is called')
        return engine.register_memory(registration.address, registration.size)

    def make_ready_event(self) -> Any:
        return torch.npu.Event()

    def close(self) -> None:
        # The TransferEngine has no explicit teardown in the mooncake API; dropping
        # the reference lets it be reclaimed when the worker shuts down.
        self._transfer_engine = None


def build_transfer_backend(device_backend: str) -> TransferBackend:
    """Construct the transfer backend for a resolved device name."""
    if device_backend == 'cuda':
        return CudaTransferBackend()
    if device_backend == 'ascend':
        return AscendTransferBackend()
    raise ValueError(f'Unsupported device_backend: {device_backend!r}')
